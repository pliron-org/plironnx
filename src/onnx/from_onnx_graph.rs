// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Build ONNX dialect IR from a topologically ordered burn-onnx graph.

use onnx_ir::{Argument, Node, OnnxGraph, ValueSource};

use pliron::{
    builtin::{
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
    result::Result,
    r#type::TypeHandle,
    utils::table::HMap,
    value::Value,
};

use crate::onnx::{
    attributes::constant_value_attr,
    ops::{AddOp, CastOp, ConstantOp, MulOp, ReturnOp},
    types::{arg_type_to_type, is_arg_element_signed},
};
use pliron_tensor::tensor::attributes::CastSignednessAttr;

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
}

/// Scoped ONNX names; inner definitions shadow outer ones.
#[derive(Default)]
pub struct ValueScopes {
    scopes: Vec<HMap<String, Value>>,
}

impl ValueScopes {
    /// Start with one empty scope.
    pub fn new() -> Self {
        ValueScopes {
            scopes: vec![HMap::default()],
        }
    }

    /// Enter a nested scope.
    pub fn push(&mut self) {
        self.scopes.push(HMap::default());
    }

    /// Leave the current nested scope.
    pub fn pop(&mut self) {
        assert!(self.scopes.len() > 1, "cannot remove the outermost scope");
        self.scopes.pop();
    }

    /// Insert in the current scope, rejecting duplicate definitions.
    pub fn insert(&mut self, name: &str, value: Value) -> Result<()> {
        let scope = self
            .scopes
            .last_mut()
            .expect("the stack always has one scope or more");
        if scope.insert(name.to_string(), value).is_some() {
            return input_err_noloc!(FromOnnxGraphErr::DuplicateValue(name.to_string()));
        }
        Ok(())
    }

    /// Look up a value from the innermost matching scope.
    pub fn lookup(&self, name: &str) -> Option<Value> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
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
    let result_types = graph
        .outputs
        .iter()
        .map(|output| arg_type_to_type(ctx, &output.ty))
        .collect::<Result<Vec<TypeHandle>>>()?;

    let func_ty = FunctionType::get(ctx, arg_types, result_types);
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

        Node::Add(add) => {
            expect_arity(node, add.inputs.len(), 2, add.outputs.len(), 1)?;
            let lhs = resolve_operand(ctx, ins, scopes, &add.inputs[0], &add.name)?;
            let rhs = resolve_operand(ctx, ins, scopes, &add.inputs[1], &add.name)?;
            let result_ty = arg_type_to_type(ctx, &add.outputs[0].ty)?;
            let op = AddOp::new(ctx, lhs, rhs, result_ty);
            let result = op.get_result(ctx);
            ins.append_op(ctx, &op);
            Ok(vec![result])
        }

        Node::Mul(mul) => {
            expect_arity(node, mul.inputs.len(), 2, mul.outputs.len(), 1)?;
            let lhs = resolve_operand(ctx, ins, scopes, &mul.inputs[0], &mul.name)?;
            let rhs = resolve_operand(ctx, ins, scopes, &mul.inputs[1], &mul.name)?;
            let result_ty = arg_type_to_type(ctx, &mul.outputs[0].ty)?;
            let op = MulOp::new(ctx, lhs, rhs, result_ty);
            let result = op.get_result(ctx);
            ins.append_op(ctx, &op);
            Ok(vec![result])
        }

        Node::Cast(cast) => {
            expect_arity(node, cast.inputs.len(), 1, cast.outputs.len(), 1)?;
            let input = resolve_operand(ctx, ins, scopes, &cast.inputs[0], &cast.name)?;
            let result_ty = arg_type_to_type(ctx, &cast.outputs[0].ty)?;
            let signedness = CastSignednessAttr {
                input_is_signed: is_arg_element_signed(&cast.inputs[0].ty),
                result_is_signed: is_arg_element_signed(&cast.outputs[0].ty),
            };
            let op = CastOp::new(ctx, input, result_ty, signedness);
            let result = op.get_result(ctx);
            ins.append_op(ctx, &op);
            Ok(vec![result])
        }

        other => input_err_noloc!(FromOnnxGraphErr::UnsupportedNode(node_kind(other))),
    }
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
        ValueSource::Constant | ValueSource::Static(_) => match scopes.lookup(&arg.name) {
            Some(value) => Ok(value),
            None => materialize_shared_constant(ctx, ins, scopes, arg),
        },

        ValueSource::Optional => input_err_noloc!(FromOnnxGraphErr::OptionalInput(
            arg.name.clone(),
            node_name.to_string()
        )),
    }
}

/// Materialize a constant argument once, and share it with all its other uses.
fn materialize_shared_constant(
    ctx: &mut Context,
    ins: &mut OpInserter,
    scopes: &mut ValueScopes,
    arg: &Argument,
) -> Result<Value> {
    let value = materialize_constant(ctx, ins, arg, arg)?;
    name_value(ctx, value, &arg.name);
    scopes.insert(&arg.name, value)?;
    Ok(value)
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
