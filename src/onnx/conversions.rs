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
        CastOpInterface, FastMathFlags, FloatBinArithOpWithFastMathFlags,
        IntBinArithOpWithOverflowFlag,
    },
    ops::{AddOp as LLVMAddOp, FAddOp, FMulOp, FPExtOp, FPToSIOp, FPTruncOp, MulOp as LLVMMulOp},
};
use pliron_tensor::{
    memref::{attributes::DenseElementsAttr, type_interfaces::ShapedType},
    tensor::{
        ops::{
            AddOp as TensorAddOp, BroadcastOp as TensorBroadcastOp, ConstantOp as TensorConstantOp,
            ElementwiseCastOp, MulOp as TensorMulOp, ReshapeOp as TensorReshapeOp, SplatOp,
        },
        types::RankedTensorType,
    },
};

use crate::onnx::ops::{AddOp, CastOp, ConstantOp, MulOp, ReshapeOp};

#[derive(Debug, thiserror::Error)]
pub enum OnnxToTensorErr {
    #[error(
        "ONNX lowering needs runtime dimensions for result type {0}, which this entry point does not yet provide"
    )]
    DynamicResultType(String),
    #[error("cannot lower ONNX {op} with operand types {lhs} and {rhs}")]
    UnsupportedBinary {
        op: &'static str,
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
            self.lhs(ctx),
            self.rhs(ctx),
            self.result_type(ctx),
            "Add",
            |ctx, rewriter, lhs, rhs| {
                let op = TensorAddOp::new(ctx, lhs, rhs);
                let result = op.get_result(ctx);
                rewriter.append_op(ctx, &op);
                result
            },
            |ctx, rewriter, lhs, rhs| {
                let op =
                    FAddOp::new_with_fast_math_flags(ctx, lhs, rhs, FastmathFlagsAttr::default());
                let result = op.get_result(ctx);
                rewriter.append_op(ctx, &op);
                result
            },
            |ctx, rewriter, lhs, rhs| {
                let op = LLVMAddOp::new_with_overflow_flag(
                    ctx,
                    lhs,
                    rhs,
                    IntegerOverflowFlagsAttr::default(),
                );
                let result = op.get_result(ctx);
                rewriter.append_op(ctx, &op);
                result
            },
        )
    }
}

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
            self.lhs(ctx),
            self.rhs(ctx),
            self.result_type(ctx),
            "Mul",
            |ctx, rewriter, lhs, rhs| {
                let op = TensorMulOp::new(ctx, lhs, rhs);
                let result = op.get_result(ctx);
                rewriter.append_op(ctx, &op);
                result
            },
            |ctx, rewriter, lhs, rhs| {
                let op =
                    FMulOp::new_with_fast_math_flags(ctx, lhs, rhs, FastmathFlagsAttr::default());
                let result = op.get_result(ctx);
                rewriter.append_op(ctx, &op);
                result
            },
            |ctx, rewriter, lhs, rhs| {
                let op = LLVMMulOp::new_with_overflow_flag(
                    ctx,
                    lhs,
                    rhs,
                    IntegerOverflowFlagsAttr::default(),
                );
                let result = op.get_result(ctx);
                rewriter.append_op(ctx, &op);
                result
            },
        )
    }
}

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
#[allow(clippy::too_many_arguments)]
fn lower_binary_op<TensorOp, FloatOp, IntegerOp>(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    source: &dyn Op,
    lhs: Value,
    rhs: Value,
    result_type: TypeHandle,
    op_name: &'static str,
    tensor_op: TensorOp,
    float_op: FloatOp,
    integer_op: IntegerOp,
) -> Result<()>
where
    TensorOp: FnOnce(&mut Context, &mut DialectConversionRewriter, Value, Value) -> Value,
    FloatOp: FnOnce(&mut Context, &mut DialectConversionRewriter, Value, Value) -> Value,
    IntegerOp: FnOnce(&mut Context, &mut DialectConversionRewriter, Value, Value) -> Value,
{
    let lhs_tensor = ranked_tensor_type(ctx, lhs);
    let rhs_tensor = ranked_tensor_type(ctx, rhs);
    let result = match (lhs_tensor, rhs_tensor) {
        (Some(_), Some(_)) => {
            let result_tensor = TypedHandle::from_handle(result_type, ctx)?;
            require_static_result(ctx, source, result_tensor)?;
            let lhs = broadcast_to_result(ctx, rewriter, lhs, result_tensor)?;
            let rhs = broadcast_to_result(ctx, rewriter, rhs, result_tensor)?;
            tensor_op(ctx, rewriter, lhs, rhs)
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
            tensor_op(ctx, rewriter, lhs, rhs)
        }
        (None, None) => {
            let lhs_type = lhs.get_type(ctx);
            let rhs_type = rhs.get_type(ctx);
            if type_impls::<dyn FloatTypeInterface>(&*lhs_type.deref(ctx))
                && type_impls::<dyn FloatTypeInterface>(&*rhs_type.deref(ctx))
            {
                float_op(ctx, rewriter, lhs, rhs)
            } else if lhs_type.deref(ctx).is::<IntegerType>()
                && rhs_type.deref(ctx).is::<IntegerType>()
            {
                integer_op(ctx, rewriter, lhs, rhs)
            } else {
                return Err(input_error!(
                    source.loc(ctx),
                    OnnxToTensorErr::UnsupportedBinary {
                        op: op_name,
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
