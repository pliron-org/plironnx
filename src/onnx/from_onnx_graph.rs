// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Build ONNX dialect IR from a topologically ordered burn-onnx graph.

use onnx_ir::{
    ArgType, Argument, DataId, Node, OnnxGraph, ValueSource, node::reshape::ReshapeInput,
};

use pliron::{
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{OneResultInterface, SingleBlockRegionInterface},
        ops::{FuncOp, ModuleOp},
        types::FunctionType,
    },
    context::Context,
    identifier::Identifier,
    input_err_noloc, input_error_noloc,
    irbuild::{
        inserter::{IRInserter, Inserter},
        listener::DummyListener,
    },
    op::Op,
    printable::Printable,
    result::Result,
    r#type::{TypeHandle, Typed, TypedHandle},
    utils::table::HMap,
    value::Value,
};

use crate::onnx::{
    attributes::{constant_value_attr, constant_value_attr_from_bytes},
    ops::{
        AddOp, CastOp, ConstantOp, DivOp, MatMulOp, MulOp, ReshapeOp, ReturnOp, SubOp,
        matmul_shapes,
    },
    types::{
        arg_type_to_type, dtype_to_type, is_arg_element_signed, is_arg_element_unsigned,
        value_shape,
    },
};
use pliron_tensor::memref::type_interfaces::{Dimension, MultiDimensionalType, ShapedType};
use pliron_tensor::tensor::attributes::CastSignednessAttr;
use pliron_tensor::tensor::types::{RankedTensorType, broadcast_shapes};

type OpInserter = IRInserter<DummyListener>;

#[derive(Debug, thiserror::Error)]
pub enum FromOnnxGraphErr {
    #[error("The ONNX operator `{0}` is not supported")]
    UnsupportedNode(String),
    #[error("The ONNX value `{0}` has no definition")]
    UndefinedValue(String),
    #[error("The ONNX value `{0}` has more than one definition in the same scope")]
    DuplicateValue(String),
    #[error("The node `{node}` has {got} input(s), but it must have {expected}")]
    NumInputs {
        node: String,
        expected: usize,
        got: usize,
    },
    #[error("The node `{node}` has {got} output(s), but it must have {expected}")]
    NumOutputs {
        node: String,
        expected: usize,
        got: usize,
    },
    #[error("The input `{0}` of the node `{1}` is optional. An optional input is not supported")]
    OptionalInput(String, String),
    #[error("The argument `{0}` must hold a constant, but it has no data")]
    MissingConstantData(String),
    #[error("The constant `{0}` must be a 2-D tensor")]
    NotAMatrix(String),
    #[error("The node `{0}` needs a compile-time reshape shape")]
    DynamicReshapeShape(String),
    #[error("The node `{0}` divides unsigned integers, which is not supported")]
    UnsignedIntegerDiv(String),
    #[error(
        "The node `{node}` declares a result extent {declared}, but its operands give {inferred}"
    )]
    ShapeConflict {
        node: String,
        declared: String,
        inferred: String,
    },
}

/// Values defined in one scope of [ValueScopes].
#[derive(Default)]
struct Scope {
    /// ONNX names mapped to their pliron values.
    names: HMap<String, Value>,
    /// Static constant data IDs mapped to their materialized pliron values.
    statics: HMap<DataId, Value>,
}

/// Scoped ONNX names and static constants.
#[derive(Default)]
pub struct ValueScopes {
    scopes: Vec<Scope>,
}

impl ValueScopes {
    /// Start with one empty scope.
    pub fn new() -> Self {
        ValueScopes {
            scopes: vec![Scope::default()],
        }
    }

    /// Enter a nested scope.
    pub fn push(&mut self) {
        self.scopes.push(Scope::default());
    }

    /// Leave the current nested scope.
    pub fn pop(&mut self) {
        assert!(self.scopes.len() > 1, "cannot remove the outermost scope");
        self.scopes.pop();
    }

    /// Insert in the current scope, rejecting duplicate definitions.
    pub fn insert(&mut self, name: &str, value: Value) -> Result<()> {
        if self
            .current()
            .names
            .insert(name.to_string(), value)
            .is_some()
        {
            return input_err_noloc!(FromOnnxGraphErr::DuplicateValue(name.to_string()));
        }
        Ok(())
    }

    /// Look up a value, starting from the innermost scope.
    pub fn lookup(&self, name: &str) -> Option<Value> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.names.get(name).copied())
    }

    /// Insert the materialized static constant `id` in the current scope.
    pub fn insert_static(&mut self, id: DataId, value: Value) {
        self.current().statics.insert(id, value);
    }

    /// Look up the static constant `id`, starting from the innermost scope.
    pub fn lookup_static(&self, id: DataId) -> Option<Value> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.statics.get(&id).copied())
    }

    /// The innermost scope.
    fn current(&mut self) -> &mut Scope {
        self.scopes
            .last_mut()
            .expect("the stack always has one scope or more")
    }
}

/// Wrap the graph in a module containing one function.
pub fn build_module(
    ctx: &mut Context,
    graph: &OnnxGraph,
    module_name: &str,
    func_name: &str,
) -> Result<ModuleOp> {
    let module = ModuleOp::new(ctx, to_identifier(module_name)?);
    let func = build_func(ctx, graph, func_name)?;
    module.append_operation(ctx, func.get_operation(), 0);
    Ok(module)
}

/// Build a function with parameters and results in graph input/output order.
pub fn build_func(ctx: &mut Context, graph: &OnnxGraph, func_name: &str) -> Result<FuncOp> {
    let arg_types = graph
        .inputs
        .iter()
        .map(|input| arg_type_to_type(ctx, &input.ty))
        .collect::<Result<Vec<TypeHandle>>>()?;
    // We don't set the result types until a little later, when the graph outputs are known.
    let func_ty = FunctionType::get(ctx, arg_types.clone(), vec![]);
    let func = FuncOp::new(ctx, to_identifier(func_name)?, func_ty);

    let entry = func.get_entry_block(ctx);
    let mut ins = OpInserter::new_at_block_end(entry);
    let mut scopes = ValueScopes::new();

    for (idx, input) in graph.inputs.iter().enumerate() {
        let arg = entry.deref(ctx).get_argument(idx);
        name_value(ctx, arg, &input.name);
        scopes.insert(&input.name, arg)?;
    }

    build_nodes(ctx, &mut ins, &mut scopes, graph)?;

    let outputs = graph
        .outputs
        .iter()
        .map(|output| resolve_operand(ctx, &mut ins, &mut scopes, output, "<graph output>"))
        .collect::<Result<Vec<Value>>>()?;
    // Compute the result types now, from the outputs' refined types.
    let result_types = outputs.iter().map(|output| output.get_type(ctx)).collect();
    let func_ty = FunctionType::get(ctx, arg_types, result_types);
    func.set_attr_builtin_func_type(ctx, TypeAttr::new(func_ty.into()));
    let ret = ReturnOp::new(ctx, outputs);
    ins.append_op(ctx, &ret);

    Ok(func)
}

/// Materialize graph nodes and record each produced value.
fn build_nodes(
    ctx: &mut Context,
    ins: &mut OpInserter,
    scopes: &mut ValueScopes,
    graph: &OnnxGraph,
) -> Result<()> {
    for node in &graph.nodes {
        let results = build_node(ctx, ins, scopes, node)?;
        for (output, result) in node.outputs().iter().zip(results) {
            name_value(ctx, result, &output.name);
            scopes.insert(&output.name, result)?;
        }
    }
    Ok(())
}

/// Build one supported ONNX node and return its results.
fn build_node(
    ctx: &mut Context,
    ins: &mut OpInserter,
    scopes: &mut ValueScopes,
    node: &Node,
) -> Result<Vec<Value>> {
    match node {
        Node::Constant(constant) => {
            expect_arity(node, constant.inputs.len(), 1, constant.outputs.len(), 1)?;
            let value = materialize_constant(ctx, ins, &constant.inputs[0], &constant.outputs[0])?;
            Ok(vec![value])
        }

        Node::Add(add) => build_binary(
            ctx,
            ins,
            scopes,
            node,
            &add.inputs,
            &add.outputs,
            AddOp::new,
        ),
        Node::Sub(sub) => build_binary(
            ctx,
            ins,
            scopes,
            node,
            &sub.inputs,
            &sub.outputs,
            SubOp::new,
        ),
        Node::Mul(mul) => build_binary(
            ctx,
            ins,
            scopes,
            node,
            &mul.inputs,
            &mul.outputs,
            MulOp::new,
        ),
        Node::Div(div) => {
            expect_arity(node, div.inputs.len(), 2, div.outputs.len(), 1)?;
            // Integer types are signless, and signless division lowers to signed division.
            if is_arg_element_unsigned(&div.outputs[0].ty) {
                return input_err_noloc!(FromOnnxGraphErr::UnsignedIntegerDiv(div.name.clone()));
            }
            build_binary(
                ctx,
                ins,
                scopes,
                node,
                &div.inputs,
                &div.outputs,
                DivOp::new,
            )
        }

        Node::MatMul(matmul) => {
            expect_arity(node, matmul.inputs.len(), 2, matmul.outputs.len(), 1)?;
            let lhs = resolve_operand(ctx, ins, scopes, &matmul.inputs[0], &matmul.name)?;
            let rhs = resolve_operand(ctx, ins, scopes, &matmul.inputs[1], &matmul.name)?;
            let declared = arg_type_to_type(ctx, &matmul.outputs[0].ty)?;
            let result = append_matmul(ctx, ins, node, lhs, rhs, declared)?;
            Ok(vec![result])
        }

        // ONNX doesn't have a `Linear` operator, but burn produces it in the following cases:
        // - A `Gemm` with `alpha = beta = 1`, `transA = 0` and `transB = 1`.
        //   Since its weight is `[out, in]` (`transB = 1`), it must be transposed.
        // - A `MatMul` with a constant 2-D weight, and an optional bias `Add` after it.
        Node::Linear(linear) => {
            let num_inputs = linear.inputs.len().clamp(2, 3);
            expect_arity(
                node,
                linear.inputs.len(),
                num_inputs,
                linear.outputs.len(),
                1,
            )?;
            let input = resolve_operand(ctx, ins, scopes, &linear.inputs[0], &linear.name)?;
            let weight = if linear.config.transpose_weight {
                materialize_transposed_constant(ctx, ins, &linear.inputs[1])?
            } else {
                resolve_operand(ctx, ins, scopes, &linear.inputs[1], &linear.name)?
            };
            let declared = arg_type_to_type(ctx, &linear.outputs[0].ty)?;
            let product = append_matmul(ctx, ins, node, input, weight, declared)?;
            let result = match linear.inputs.get(2) {
                Some(bias) if !matches!(bias.value_source, ValueSource::Optional) => {
                    let bias = resolve_operand(ctx, ins, scopes, bias, &linear.name)?;
                    append_binary(ctx, ins, node, product, bias, declared, AddOp::new)?
                }
                _ => product,
            };
            Ok(vec![result])
        }

        Node::Cast(cast) => {
            expect_arity(node, cast.inputs.len(), 1, cast.outputs.len(), 1)?;
            let input = resolve_operand(ctx, ins, scopes, &cast.inputs[0], &cast.name)?;
            let declared = arg_type_to_type(ctx, &cast.outputs[0].ty)?;
            let inferred = value_shape(ctx, input.get_type(ctx));
            let result_ty = refine_result_type(ctx, node, declared, inferred)?;
            let signedness = CastSignednessAttr {
                input_is_signed: is_arg_element_signed(&cast.inputs[0].ty),
                result_is_signed: is_arg_element_signed(&cast.outputs[0].ty),
            };
            let op = CastOp::new(ctx, input, result_ty, signedness);
            let result = op.get_result(ctx);
            ins.append_op(ctx, &op);
            Ok(vec![result])
        }

        Node::Reshape(reshape) => {
            expect_arity(node, reshape.inputs.len(), 2, reshape.outputs.len(), 1)?;
            if !matches!(reshape.config.shape, ReshapeInput::Static(_)) {
                return input_err_noloc!(FromOnnxGraphErr::DynamicReshapeShape(
                    reshape.name.clone()
                ));
            }
            let input = resolve_operand(ctx, ins, scopes, &reshape.inputs[0], &reshape.name)?;
            let result_ty = TypedHandle::<RankedTensorType>::from_handle(
                arg_type_to_type(ctx, &reshape.outputs[0].ty)?,
                ctx,
            )?;
            let op = ReshapeOp::new(ctx, input, result_ty);
            let result = op.get_result(ctx);
            ins.append_op(ctx, &op);
            Ok(vec![result])
        }

        other => input_err_noloc!(FromOnnxGraphErr::UnsupportedNode(node_kind(other))),
    }
}

/// Build a broadcasting binary operation with one result.
fn build_binary<T: Op>(
    ctx: &mut Context,
    ins: &mut OpInserter,
    scopes: &mut ValueScopes,
    node: &Node,
    inputs: &[Argument],
    outputs: &[Argument],
    new_op: fn(&mut Context, Value, Value, TypeHandle) -> T,
) -> Result<Vec<Value>> {
    expect_arity(node, inputs.len(), 2, outputs.len(), 1)?;
    let lhs = resolve_operand(ctx, ins, scopes, &inputs[0], node.name())?;
    let rhs = resolve_operand(ctx, ins, scopes, &inputs[1], node.name())?;
    let declared = arg_type_to_type(ctx, &outputs[0].ty)?;
    let result = append_binary(ctx, ins, node, lhs, rhs, declared, new_op)?;
    Ok(vec![result])
}

/// Append a broadcasting binary operation on `lhs` and `rhs`, and return its result.
///
/// The `declared` result type is refined with the broadcast shape of the operands.
fn append_binary<T: Op>(
    ctx: &mut Context,
    ins: &mut OpInserter,
    node: &Node,
    lhs: Value,
    rhs: Value,
    declared: TypeHandle,
    new_op: fn(&mut Context, Value, Value, TypeHandle) -> T,
) -> Result<Value> {
    let inferred = value_shape(ctx, lhs.get_type(ctx))
        .zip(value_shape(ctx, rhs.get_type(ctx)))
        .and_then(|(lhs, rhs)| broadcast_shapes(&lhs, &rhs));
    let result_ty = refine_result_type(ctx, node, declared, inferred)?;
    let op = new_op(ctx, lhs, rhs, result_ty);
    let result = op.get_operation().deref(ctx).get_result(0);
    ins.append_op(ctx, &op);
    Ok(result)
}

/// Append the matrix product of `lhs` and `rhs`, and return its result.
///
/// The `declared` result type is refined with the product shape of the operands.
fn append_matmul(
    ctx: &mut Context,
    ins: &mut OpInserter,
    node: &Node,
    lhs: Value,
    rhs: Value,
    declared: TypeHandle,
) -> Result<Value> {
    let inferred = value_shape(ctx, lhs.get_type(ctx))
        .zip(value_shape(ctx, rhs.get_type(ctx)))
        .and_then(|(lhs, rhs)| matmul_shapes(&lhs, &rhs))
        .map(|shapes| shapes.result);
    let result_ty = TypedHandle::<RankedTensorType>::from_handle(
        refine_result_type(ctx, node, declared, inferred)?,
        ctx,
    )?;
    let op = MatMulOp::new(ctx, lhs, rhs, result_ty);
    let result = op.get_result(ctx);
    ins.append_op(ctx, &op);
    Ok(result)
}

/// Refine the `declared` tensor type of a node result with its `inferred` shape.
///
/// Each extent is computed as `declared ⊓ inferred`,
/// where `?` is ⊤ and `n ⊓ m` is ⊥ (an error) for `n ≠ m`.
///
/// If `declared` is not a tensor or has a different rank than `inferred`,
/// it is returned unchanged.
fn refine_result_type(
    ctx: &Context,
    node: &Node,
    declared: TypeHandle,
    inferred: Option<Vec<Dimension>>,
) -> Result<TypeHandle> {
    let Ok(declared_tensor) = TypedHandle::<RankedTensorType>::from_handle(declared, ctx) else {
        return Ok(declared);
    };
    let (element_type, declared_shape) = {
        let tensor = declared_tensor.deref(ctx);
        (tensor.element_type(), tensor.shape().clone())
    };
    let Some(inferred) = inferred.filter(|inferred| inferred.len() == declared_shape.len()) else {
        return Ok(declared);
    };

    let shape = declared_shape
        .iter()
        .zip(&inferred)
        .map(|(declared, inferred)| match (declared, inferred) {
            (Dimension::Static(d), Dimension::Static(i)) if d != i => {
                input_err_noloc!(FromOnnxGraphErr::ShapeConflict {
                    node: node.name().to_string(),
                    declared: declared.disp(ctx).to_string(),
                    inferred: inferred.disp(ctx).to_string(),
                })
            }
            (Dimension::Dynamic, inferred) => Ok(inferred.clone()),
            (declared, _) => Ok(declared.clone()),
        })
        .collect::<Result<Vec<Dimension>>>()?;
    Ok(RankedTensorType::get(ctx, element_type, shape).into())
}

/// Resolve an argument to an existing or newly materialized SSA value.
fn resolve_operand(
    ctx: &mut Context,
    ins: &mut OpInserter,
    scopes: &mut ValueScopes,
    arg: &Argument,
    node_name: &str,
) -> Result<Value> {
    match arg.value_source {
        ValueSource::Dynamic => scopes
            .lookup(&arg.name)
            .ok_or_else(|| input_error_noloc!(FromOnnxGraphErr::UndefinedValue(arg.name.clone()))),

        // Constant nodes may have been folded away; fall back to the argument data.
        // Materialize each constant once, and share it with all its other uses.
        ValueSource::Constant => match scopes.lookup(&arg.name) {
            Some(value) => Ok(value),
            None => {
                let value = materialize_constant(ctx, ins, arg, arg)?;
                name_value(ctx, value, &arg.name);
                scopes.insert(&arg.name, value)?;
                Ok(value)
            }
        },

        ValueSource::Static(id) => match scopes.lookup_static(id) {
            Some(value) => Ok(value),
            None => {
                let value = materialize_constant(ctx, ins, arg, arg)?;
                scopes.insert_static(id, value);
                Ok(value)
            }
        },

        ValueSource::Optional => input_err_noloc!(FromOnnxGraphErr::OptionalInput(
            arg.name.clone(),
            node_name.to_string()
        )),
    }
}

/// Constant nodes carry data on the input and the result type on the output.
fn materialize_constant(
    ctx: &mut Context,
    ins: &mut OpInserter,
    data_arg: &Argument,
    result_arg: &Argument,
) -> Result<Value> {
    let data = data_arg.value().ok_or_else(|| {
        input_error_noloc!(FromOnnxGraphErr::MissingConstantData(data_arg.name.clone()))
    })?;
    let result_ty = arg_type_to_type(ctx, &result_arg.ty)?;
    let value_attr = constant_value_attr(ctx, &data, result_ty)?;

    let op = ConstantOp::new(ctx, value_attr);
    let result = op.get_result(ctx);
    ins.append_op(ctx, &op);
    Ok(result)
}

/// Swap `arg`'s two dimensions into a new result.
fn materialize_transposed_constant(
    ctx: &mut Context,
    ins: &mut OpInserter,
    arg: &Argument,
) -> Result<Value> {
    let data = arg.value().ok_or_else(|| {
        input_error_noloc!(FromOnnxGraphErr::MissingConstantData(arg.name.clone()))
    })?;
    let element_size = data.dtype().size();
    let bytes = data.as_bytes();
    let &[rows, cols] = &data.shape()[..] else {
        return input_err_noloc!(FromOnnxGraphErr::NotAMatrix(arg.name.clone()));
    };
    if bytes.len() != rows * cols * element_size {
        return input_err_noloc!(FromOnnxGraphErr::MissingConstantData(arg.name.clone()));
    }

    let mut transposed = Vec::with_capacity(bytes.len());
    for col in 0..cols {
        for row in 0..rows {
            let start = (row * cols + col) * element_size;
            transposed.extend_from_slice(&bytes[start..start + element_size]);
        }
    }

    let ArgType::Tensor(tensor_ty) = &arg.ty else {
        return input_err_noloc!(FromOnnxGraphErr::NotAMatrix(arg.name.clone()));
    };
    let element_type = dtype_to_type(ctx, tensor_ty.dtype)?;
    let result_ty: TypeHandle = RankedTensorType::get(
        ctx,
        element_type,
        vec![Dimension::Static(cols), Dimension::Static(rows)],
    )
    .into();
    let value_attr = constant_value_attr_from_bytes(ctx, data.dtype(), transposed, result_ty)?;

    let op = ConstantOp::new(ctx, value_attr);
    let result = op.get_result(ctx);
    ins.append_op(ctx, &op);
    Ok(result)
}

/// Convert an ONNX name to a pliron identifier.
fn to_identifier(name: &str) -> Result<Identifier> {
    Identifier::try_from(name).map_err(|err| input_error_noloc!(err))
}

/// Preserve ONNX names that are valid pliron identifiers.
fn name_value(ctx: &Context, value: Value, name: &str) {
    if let Ok(identifier) = Identifier::try_from(name) {
        value.set_name(ctx, Some(identifier));
    }
}

/// Validate a node's input and output counts.
fn expect_arity(
    node: &Node,
    num_inputs: usize,
    expected_inputs: usize,
    num_outputs: usize,
    expected_outputs: usize,
) -> Result<()> {
    if num_inputs != expected_inputs {
        return input_err_noloc!(FromOnnxGraphErr::NumInputs {
            node: node.name().to_string(),
            expected: expected_inputs,
            got: num_inputs,
        });
    }
    if num_outputs != expected_outputs {
        return input_err_noloc!(FromOnnxGraphErr::NumOutputs {
            node: node.name().to_string(),
            expected: expected_outputs,
            got: num_outputs,
        });
    }
    Ok(())
}

/// NodeType is private; extract the operator and name from Display output.
fn node_kind(node: &Node) -> String {
    node.to_string()
        .lines()
        .next()
        .unwrap_or_default()
        .trim()
        .to_string()
}
