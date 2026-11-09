// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Lower ONNX operations to tensor and LLVM operations.

use pliron::{
    attribute::boxed_attr_cast,
    builtin::{
        op_interfaces::{OneOpdInterface, OneResultInterface},
        ops::ConstantOp as BuiltinConstantOp,
        type_interfaces::FloatTypeInterface,
        types::IntegerType,
    },
    context::{Context, Ptr},
    derive::{op_interface, op_interface_impl},
    input_error,
    irbuild::{
        dialect_conversion::{DialectConversion, DialectConversionRewriter, OperandsInfo},
        inserter::Inserter,
        rewriter::Rewriter,
    },
    op::{Op, op_cast, op_impls},
    operation::Operation,
    printable::Printable,
    result::Result,
    r#type::{TypeHandle, Typed, TypedHandle, type_cast, type_impls},
    value::Value,
};
use pliron_llvm::{
    attributes::{FastmathFlagsAttr, IntegerOverflowFlagsAttr},
    op_interfaces::{
        BinArithOp, CastOpInterface, FastMathFlags, FloatBinArithOpWithFastMathFlags,
        IntBinArithOpWithOverflowFlag,
    },
    ops::{
        AddOp as LLVMAddOp, FAddOp, FDivOp, FMulOp, FPExtOp, FPToSIOp, FPTruncOp, FSubOp,
        MulOp as LLVMMulOp, SDivOp, SubOp as LLVMSubOp,
    },
};
use pliron_tensor::{
    memref::{
        attributes::DenseElementsAttr,
        type_interfaces::{DenseElementTypeHandle, Dimension, MultiDimensionalType, ShapedType},
    },
    tensor::{
        ops::{
            AddOp as TensorAddOp, BatchMatMulOp, BroadcastOp as TensorBroadcastOp,
            ConstantOp as TensorConstantOp, DivOp as TensorDivOp, ElementwiseCastOp,
            MatMulOp as TensorMatMulOp, MulOp as TensorMulOp, ReshapeOp as TensorReshapeOp,
            SplatOp, SubOp as TensorSubOp,
        },
        types::RankedTensorType,
    },
};

use crate::onnx::ops::{
    AddOp, BroadcastableBinaryOpInterface, CastOp, ConstantOp, DivOp, MatMulOp, MulOp, ReshapeOp,
    SubOp, matmul_shapes,
};

#[derive(Debug, thiserror::Error)]
pub enum OnnxToTensorErr {
    #[error(
        "ONNX lowering needs runtime dimensions for result type {0}, which this entry point does not yet provide"
    )]
    DynamicResultType(String),
    #[error("ONNX lowering does not support the dynamic operand type {0}")]
    DynamicOperandType(String),
    #[error("cannot lower {op} with operand types {lhs} and {rhs}")]
    UnsupportedBinary {
        op: String,
        lhs: String,
        rhs: String,
    },
    #[error("cannot lower ONNX cast from {from} to {to}")]
    UnsupportedCast { from: String, to: String },
}

enum ScalarCastKind {
    TruncateFloat,
    ExtendFloat,
    FloatToSignedInteger,
}

/// The conversion from the `onnx` dialect to the `tensor` dialect.
pub struct OnnxToTensor;

/// An ONNX operation that can lower itself to tensor and LLVM operations.
#[op_interface]
trait ToTensorDialect {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()>;

    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// Lowers a tensor constant to `tensor.constant` and a scalar constant to `builtin.constant`.
///
/// ```text
/// %t = onnx.constant <dense<[1, 2, 3, 4]>> : tensor<2x2xi32>
/// %s = onnx.constant <1.5> : f32
/// ```
///
/// becomes:
///
/// ```text
/// %t = tensor.constant dense<[1, 2, 3, 4]> : tensor<2x2xi32>
/// %s = builtin.constant 1.5 : f32
/// ```
#[op_interface_impl]
impl ToTensorDialect for ConstantOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        // `take_value` makes `self` incorrect, but `self` is replaced below.
        let value = self.take_value(ctx);
        let new_op = match value.downcast::<DenseElementsAttr>() {
            Ok(dense) => {
                let op = TensorConstantOp::new(ctx, *dense);
                rewriter.append_op(ctx, &op);
                op.get_operation()
            }
            Err(value) => {
                let value = boxed_attr_cast(value)
                    .expect("onnx.constant value attribute must implement TypedAttrInterface");
                let op = BuiltinConstantOp::new(ctx, value);
                rewriter.append_op(ctx, &op);
                op.get_operation()
            }
        };
        rewriter.replace_operation(ctx, self.get_operation(), new_op);
        Ok(())
    }
}

/// Lowers `onnx.add` to `tensor.add`, or to `llvm.fadd` / `llvm.add` for two scalars.
/// See [lower_binary_op] for the broadcast of the operands.
///
/// ```text
/// %r = onnx.add %a, %b : tensor<2x3xf32>
/// %x = onnx.add %s, %t : i32
/// ```
///
/// becomes:
///
/// ```text
/// %r = tensor.add %a, %b : tensor<2x3xf32>
/// %x = llvm.add %s, %t : i32
/// ```
#[op_interface_impl]
impl ToTensorDialect for AddOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        lower_binary_op(
            ctx,
            rewriter,
            self,
            TensorAddOp::new,
            |ctx, lhs, rhs| {
                FAddOp::new_with_fast_math_flags(ctx, lhs, rhs, FastmathFlagsAttr::default())
            },
            |ctx, lhs, rhs| {
                LLVMAddOp::new_with_overflow_flag(
                    ctx,
                    lhs,
                    rhs,
                    IntegerOverflowFlagsAttr::default(),
                )
            },
        )
    }
}

/// Lowers `onnx.mul` to `tensor.mul`, or to `llvm.fmul` / `llvm.mul` for two scalars.
/// See [lower_binary_op] for the broadcast of the operands.
///
/// ```text
/// %r = onnx.mul %a, %b : tensor<2x3xf32>
/// %x = onnx.mul %s, %t : f32
/// ```
///
/// becomes:
///
/// ```text
/// %r = tensor.mul %a, %b : tensor<2x3xf32>
/// %x = llvm.fmul %s, %t : f32
/// ```
#[op_interface_impl]
impl ToTensorDialect for MulOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        lower_binary_op(
            ctx,
            rewriter,
            self,
            TensorMulOp::new,
            |ctx, lhs, rhs| {
                FMulOp::new_with_fast_math_flags(ctx, lhs, rhs, FastmathFlagsAttr::default())
            },
            |ctx, lhs, rhs| {
                LLVMMulOp::new_with_overflow_flag(
                    ctx,
                    lhs,
                    rhs,
                    IntegerOverflowFlagsAttr::default(),
                )
            },
        )
    }
}

/// Lowers `onnx.sub` to `tensor.sub`, or to `llvm.fsub` / `llvm.sub` for two scalars.
/// See [lower_binary_op] for the broadcast of the operands.
///
/// ```text
/// %r = onnx.sub %a, %b : tensor<2x3xf32>
/// %x = onnx.sub %s, %t : i32
/// ```
///
/// becomes:
///
/// ```text
/// %r = tensor.sub %a, %b : tensor<2x3xf32>
/// %x = llvm.sub %s, %t : i32
/// ```
#[op_interface_impl]
impl ToTensorDialect for SubOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        lower_binary_op(
            ctx,
            rewriter,
            self,
            TensorSubOp::new,
            |ctx, lhs, rhs| {
                FSubOp::new_with_fast_math_flags(ctx, lhs, rhs, FastmathFlagsAttr::default())
            },
            |ctx, lhs, rhs| {
                LLVMSubOp::new_with_overflow_flag(
                    ctx,
                    lhs,
                    rhs,
                    IntegerOverflowFlagsAttr::default(),
                )
            },
        )
    }
}

/// Lowers `onnx.div` to `tensor.div`, or to `llvm.fdiv` / `llvm.sdiv` for two scalars.
/// See [lower_binary_op] for the broadcast of the operands.
/// Integer division is signed, because the import rejects unsigned division.
///
/// ```text
/// %r = onnx.div %a, %b : tensor<2x3xf32>
/// %x = onnx.div %s, %t : i32
/// ```
///
/// becomes:
///
/// ```text
/// %r = tensor.div %a, %b : tensor<2x3xf32>
/// %x = llvm.sdiv %s, %t : i32
/// ```
#[op_interface_impl]
impl ToTensorDialect for DivOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        lower_binary_op(
            ctx,
            rewriter,
            self,
            TensorDivOp::new,
            |ctx, lhs, rhs| {
                FDivOp::new_with_fast_math_flags(ctx, lhs, rhs, FastmathFlagsAttr::default())
            },
            // The import rejects unsigned integer division, so the operands are signed.
            SDivOp::new,
        )
    }
}

/// Lowers `onnx.cast` of:
/// - a tensor to `tensor.elementwise_cast`
/// - a scalar to `llvm.fptrunc`, `llvm.fpext` or `llvm.fptosi`
/// - a scalar to the same type to its input
///
/// ```text
/// %r = onnx.cast %t to tensor<2x1xi32> {input_is_signed=false,result_is_signed=true}
/// %y = onnx.cast %x to i32 {input_is_signed=false,result_is_signed=true}   // %x: f32
/// ```
///
/// becomes:
///
/// ```text
/// %r = tensor.elementwise_cast %t : tensor<2x1xi32> {input_is_signed=false,result_is_signed=true}
/// %y = llvm.fptosi %x to i32
/// ```
#[op_interface_impl]
impl ToTensorDialect for CastOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let input = self.get_operand(ctx);
        let result_type = self.result_type(ctx);
        let result = if ranked_tensor_type(ctx, input).is_some() {
            let result_tensor = TypedHandle::from_handle(result_type, ctx)?;
            let op = ElementwiseCastOp::new(ctx, input, result_tensor, self.signedness(ctx));
            let result = op.get_result(ctx);
            rewriter.append_op(ctx, &op);
            result
        } else {
            let input_type = input.get_type(ctx);
            if input_type == result_type {
                input
            } else {
                let cast_kind = {
                    let input_type_ref = input_type.deref(ctx);
                    let result_type_ref = result_type.deref(ctx);
                    let input_float = type_cast::<dyn FloatTypeInterface>(&*input_type_ref);
                    let result_float = type_cast::<dyn FloatTypeInterface>(&*result_type_ref);
                    match (input_float, result_float) {
                        (Some(input_float), Some(result_float))
                            if input_float.get_semantics().bits
                                > result_float.get_semantics().bits =>
                        {
                            ScalarCastKind::TruncateFloat
                        }
                        (Some(input_float), Some(result_float))
                            if input_float.get_semantics().bits
                                < result_float.get_semantics().bits =>
                        {
                            ScalarCastKind::ExtendFloat
                        }
                        (Some(_), None)
                            if result_type_ref.is::<IntegerType>()
                                && self.signedness(ctx).result_is_signed =>
                        {
                            ScalarCastKind::FloatToSignedInteger
                        }
                        _ => {
                            return Err(input_error!(
                                self.loc(ctx),
                                OnnxToTensorErr::UnsupportedCast {
                                    from: input_type.disp(ctx).to_string(),
                                    to: result_type.disp(ctx).to_string(),
                                }
                            ));
                        }
                    }
                };
                match cast_kind {
                    ScalarCastKind::TruncateFloat => {
                        let op = FPTruncOp::new(ctx, input, result_type);
                        op.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
                        let result = op.get_result(ctx);
                        rewriter.append_op(ctx, &op);
                        result
                    }
                    ScalarCastKind::ExtendFloat => {
                        let op = FPExtOp::new(ctx, input, result_type);
                        op.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
                        let result = op.get_result(ctx);
                        rewriter.append_op(ctx, &op);
                        result
                    }
                    ScalarCastKind::FloatToSignedInteger => {
                        let op = FPToSIOp::new(ctx, input, result_type);
                        let result = op.get_result(ctx);
                        rewriter.append_op(ctx, &op);
                        result
                    }
                }
            }
        };
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![result]);
        Ok(())
    }
}

/// Lowers `onnx.reshape` with a static result shape to `tensor.reshape`.
///
/// ```text
/// %r = onnx.reshape %t to tensor<2x3xf32>   // %t: tensor<6xf32>
/// ```
///
/// becomes:
///
/// ```text
/// %r = tensor.reshape %t : tensor<2x3xf32>
/// ```
#[op_interface_impl]
impl ToTensorDialect for ReshapeOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let result_type = TypedHandle::from_handle(self.result_type(ctx), ctx)?;
        require_static_result(ctx, self, result_type)?;
        let op = TensorReshapeOp::new(ctx, self.get_operand(ctx), vec![], result_type);
        rewriter.append_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![op.get_result(ctx)]);
        Ok(())
    }
}

/// Lowers `onnx.matmul` in these steps:
/// - reshape vector operands to matrices
/// - broadcast the batch dimensions
/// - multiply into a zero accumulator with `tensor.matmul` (rank 2) or `tensor.batch_matmul`
/// - reshape to remove the dimensions that vector operands add
///
/// ```text
/// %r = onnx.matmul %v, %m : tensor<2x4xf32>   // %v: tensor<3xf32>, %m: tensor<2x3x4xf32>
/// ```
///
/// becomes:
///
/// ```text
/// %v1 = tensor.reshape %v : tensor<1x3xf32>
/// %v2 = tensor.broadcast %v1 : tensor<2x1x3xf32>
/// %zero = builtin.constant 0.0 : f32
/// %acc = tensor.splat %zero : tensor<2x1x4xf32>
/// %p = tensor.batch_matmul %v2, %m, %acc : tensor<2x1x4xf32>
/// %r = tensor.reshape %p : tensor<2x4xf32>
/// ```
#[op_interface_impl]
impl ToTensorDialect for MatMulOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let result_type = TypedHandle::<RankedTensorType>::from_handle(self.result_type(ctx), ctx)?;
        require_static_result(ctx, self, result_type)?;
        let (lhs, rhs) = (self.lhs(ctx), self.rhs(ctx));
        let lhs_type = ranked_tensor_type(ctx, lhs).expect("onnx.matmul operands are tensors");
        let rhs_type = ranked_tensor_type(ctx, rhs).expect("onnx.matmul operands are tensors");
        // The lowering builds its intermediate shapes from the operand shapes.
        require_static_operand(ctx, self, lhs_type)?;
        require_static_operand(ctx, self, rhs_type)?;
        let shapes = matmul_shapes(lhs_type.deref(ctx).shape(), rhs_type.deref(ctx).shape())
            .expect("the verifier checks that the onnx.matmul shapes are compatible");
        let element_type = result_type.deref(ctx).element_type();
        let tensor_type = |ctx: &Context, shape: &[Dimension]| {
            RankedTensorType::get(ctx, element_type, shape.to_vec())
        };

        // Promote vectors to matrices, then give both operands the same batch dimensions.
        let lhs_matrix = tensor_type(ctx, &shapes.lhs_matrix);
        let lhs = reshape_to(ctx, rewriter, lhs, lhs_matrix);
        let lhs_batched = tensor_type(ctx, &shapes.lhs_batched);
        let lhs = broadcast_to_result(ctx, rewriter, lhs, lhs_batched)?;
        let rhs_matrix = tensor_type(ctx, &shapes.rhs_matrix);
        let rhs = reshape_to(ctx, rewriter, rhs, rhs_matrix);
        let rhs_batched = tensor_type(ctx, &shapes.rhs_batched);
        let rhs = broadcast_to_result(ctx, rewriter, rhs, rhs_batched)?;

        // The tensor matrix products add to an accumulator, which is zero here.
        let result_batched_type = tensor_type(ctx, &shapes.result_batched);
        let zero = zero_scalar(ctx, rewriter, element_type)?;
        let accum = SplatOp::new(ctx, zero, vec![], result_batched_type);
        rewriter.append_op(ctx, &accum);
        let accum = accum.get_result(ctx);

        let result_batched = if shapes.result_batched.len() == 2 {
            let op = TensorMatMulOp::new(ctx, lhs, rhs, accum);
            rewriter.append_op(ctx, &op);
            op.get_result(ctx)
        } else {
            let op = BatchMatMulOp::new(ctx, lhs, rhs, accum);
            rewriter.append_op(ctx, &op);
            op.get_result(ctx)
        };

        // Remove the dimensions that vector operands add.
        let result = reshape_to(ctx, rewriter, result_batched, result_type);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![result]);
        Ok(())
    }
}

impl DialectConversion for OnnxToTensor {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_impls::<dyn ToTensorDialect>(&*Operation::get_op_dyn(op, ctx))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op_dyn = Operation::get_op_dyn(op, ctx);
        let to_tensor_op = op_cast::<dyn ToTensorDialect>(&*op_dyn)
            .expect("can_convert_op matches operations implementing ToTensorDialect");
        to_tensor_op.rewrite(ctx, rewriter, operands_info)
    }
}

/// Lower an ONNX binary operation whose tensor, floating-point scalar, and
/// integer scalar variants differ only in the operation they construct.
///
/// - Two tensors: broadcast both to the result shape.
/// - A tensor and a scalar: broadcast the tensor and splat the scalar.
/// - Two scalars: use `float_op` or `integer_op`.
///
/// ```text
/// %r = onnx.add %a, %b : tensor<2x3xf32>   // %a: tensor<2x3xf32>, %b: tensor<3xf32>
/// %q = onnx.add %a, %s : tensor<2x3xf32>   // %s: f32
/// %x = onnx.add %s, %t : f32
/// ```
///
/// becomes:
///
/// ```text
/// %b1 = tensor.broadcast %b : tensor<2x3xf32>
/// %r = tensor.add %a, %b1 : tensor<2x3xf32>
/// %s1 = tensor.splat %s : tensor<2x3xf32>
/// %q = tensor.add %a, %s1 : tensor<2x3xf32>
/// %x = llvm.fadd %s, %t : f32
/// ```
fn lower_binary_op<TensorOp: Op, FloatOp: Op, IntegerOp: Op>(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    source: &dyn BroadcastableBinaryOpInterface,
    tensor_op: fn(&mut Context, Value, Value) -> TensorOp,
    float_op: fn(&mut Context, Value, Value) -> FloatOp,
    integer_op: fn(&mut Context, Value, Value) -> IntegerOp,
) -> Result<()> {
    let (lhs, rhs) = (source.lhs(ctx), source.rhs(ctx));
    let result_type = source.result_type(ctx);
    let lhs_tensor = ranked_tensor_type(ctx, lhs);
    let rhs_tensor = ranked_tensor_type(ctx, rhs);
    let result = match (lhs_tensor, rhs_tensor) {
        (Some(_), Some(_)) => {
            let result_tensor = TypedHandle::from_handle(result_type, ctx)?;
            require_static_result(ctx, source, result_tensor)?;
            let lhs = broadcast_to_result(ctx, rewriter, lhs, result_tensor)?;
            let rhs = broadcast_to_result(ctx, rewriter, rhs, result_tensor)?;
            append_binary(ctx, rewriter, tensor_op, lhs, rhs)
        }
        (Some(_), None) | (None, Some(_)) => {
            let result_tensor = TypedHandle::from_handle(result_type, ctx)?;
            require_static_result(ctx, source, result_tensor)?;
            let (tensor, scalar) = if lhs_tensor.is_some() {
                (lhs, rhs)
            } else {
                (rhs, lhs)
            };
            let tensor = broadcast_to_result(ctx, rewriter, tensor, result_tensor)?;
            let splat = SplatOp::new(ctx, scalar, vec![], result_tensor);
            let scalar_tensor = splat.get_result(ctx);
            rewriter.append_op(ctx, &splat);
            // Keep the operand order of the source operation.
            let (lhs, rhs) = if lhs_tensor.is_some() {
                (tensor, scalar_tensor)
            } else {
                (scalar_tensor, tensor)
            };
            append_binary(ctx, rewriter, tensor_op, lhs, rhs)
        }
        (None, None) => {
            let lhs_type = lhs.get_type(ctx);
            let rhs_type = rhs.get_type(ctx);
            if type_impls::<dyn FloatTypeInterface>(&*lhs_type.deref(ctx))
                && type_impls::<dyn FloatTypeInterface>(&*rhs_type.deref(ctx))
            {
                append_binary(ctx, rewriter, float_op, lhs, rhs)
            } else if lhs_type.deref(ctx).is::<IntegerType>()
                && rhs_type.deref(ctx).is::<IntegerType>()
            {
                append_binary(ctx, rewriter, integer_op, lhs, rhs)
            } else {
                return Err(input_error!(
                    source.loc(ctx),
                    OnnxToTensorErr::UnsupportedBinary {
                        op: source.get_opid().to_string(),
                        lhs: lhs_type.disp(ctx).to_string(),
                        rhs: rhs_type.disp(ctx).to_string(),
                    }
                ));
            }
        }
    };
    rewriter.replace_operation_with_values(ctx, source.get_operation(), vec![result]);
    Ok(())
}

/// Build the binary operation `new_op(lhs, rhs)`, append it, and return its result.
fn append_binary<T: Op>(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    new_op: fn(&mut Context, Value, Value) -> T,
    lhs: Value,
    rhs: Value,
) -> Value {
    let op = new_op(ctx, lhs, rhs);
    let result = op.get_operation().deref(ctx).get_result(0);
    rewriter.append_op(ctx, &op);
    result
}

fn ranked_tensor_type(
    ctx: &Context,
    value: pliron::value::Value,
) -> Option<TypedHandle<RankedTensorType>> {
    TypedHandle::from_handle(value.get_type(ctx), ctx).ok()
}

fn require_static_result(
    ctx: &Context,
    op: &dyn Op,
    result: TypedHandle<RankedTensorType>,
) -> Result<()> {
    if result.deref(ctx).num_dynamic_dimensions() != 0 {
        let result_type: TypeHandle = result.into();
        return Err(input_error!(
            op.loc(ctx),
            OnnxToTensorErr::DynamicResultType(result_type.disp(ctx).to_string())
        ));
    }
    Ok(())
}

fn require_static_operand(
    ctx: &Context,
    op: &dyn Op,
    operand: TypedHandle<RankedTensorType>,
) -> Result<()> {
    if operand.deref(ctx).num_dynamic_dimensions() != 0 {
        let operand_type: TypeHandle = operand.into();
        return Err(input_error!(
            op.loc(ctx),
            OnnxToTensorErr::DynamicOperandType(operand_type.disp(ctx).to_string())
        ));
    }
    Ok(())
}

fn broadcast_to_result(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: pliron::value::Value,
    result_type: TypedHandle<RankedTensorType>,
) -> Result<pliron::value::Value> {
    let value_type = ranked_tensor_type(ctx, value).expect("caller supplies a tensor");
    if value_type == result_type {
        return Ok(value);
    }
    let broadcast = TensorBroadcastOp::new(ctx, value, vec![], result_type);
    let result = broadcast.get_result(ctx);
    rewriter.append_op(ctx, &broadcast);
    Ok(result)
}

/// Reshape `value` to `result_type`.
fn reshape_to(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: Value,
    result_type: TypedHandle<RankedTensorType>,
) -> Value {
    if ranked_tensor_type(ctx, value) == Some(result_type) {
        return value;
    }
    let reshape = TensorReshapeOp::new(ctx, value, vec![], result_type);
    rewriter.append_op(ctx, &reshape);
    reshape.get_result(ctx)
}

/// Build a builtin constant zero of the scalar type `element_type`.
fn zero_scalar(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    element_type: TypeHandle,
) -> Result<Value> {
    let zero = {
        let element = DenseElementTypeHandle::from_handle(element_type, ctx)?;
        let element = element.deref(ctx);
        element.element_attr(ctx, &vec![0u8; element.element_size()])
    };
    let op = BuiltinConstantOp::new(ctx, zero);
    rewriter.append_op(ctx, &op);
    Ok(op.get_result(ctx))
}
