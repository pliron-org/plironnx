// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! ONNX dialect operations.

use core::cell::Ref;
use pliron::{
    attribute::{AttrObj, attr_cast},
    builtin::{
        attr_interfaces::TypedAttrInterface,
        op_interfaces::{
            IsTerminatorInterface, NOpdsInterface, NResultsInterface, OneOpdInterface,
            OneResultInterface,
        },
    },
    common_traits::Verify,
    context::Context,
    derive::{op_interface, pliron_op},
    op::{Op, op_cast},
    operation::Operation,
    printable::Printable,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
    verify_err, verify_error,
};
use pliron_tensor::{
    memref::type_interfaces::Dimension,
    tensor::{attributes::CastSignednessAttr, types::broadcast_shapes},
};

use crate::onnx::types::{value_element_type, value_shape};

/// ONNX constant or initializer with a typed tensor or scalar value attribute.
///
/// ### Result(s):
///
/// | result | description |
/// |-----|-------|
/// | `result` | The constant tensor or scalar. |
#[pliron_op(
    name = "onnx.constant",
    format = "`<` $onnx_constant_value `>` ` : ` type($0)",
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (onnx_constant_value),
)]
pub struct ConstantOp;

#[derive(Debug, thiserror::Error)]
pub enum ConstantOpVerifyErr {
    #[error("onnx.constant does not have a value attribute")]
    MissingValue,
    #[error("The value attribute {0} of onnx.constant does not implement TypedAttrInterface")]
    ValueNotTyped(String),
    #[error("onnx.constant of type {result} has a value attribute of type {value}")]
    ValueTypeMismatch { result: String, value: String },
}

impl ConstantOp {
    /// Create a new ONNX constant.
    pub fn new(ctx: &mut Context, value: Box<dyn TypedAttrInterface>) -> Self {
        let result_type = value.get_type(ctx);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![],
            vec![],
            0,
        );
        let op = ConstantOp { op };
        op.set_attr_onnx_constant_value(ctx, value);
        op
    }

    /// Get the constant value that this operation defines.
    /// The [Ref] is a borrow of the containing [Operation] object.
    ///
    /// Use [`pliron::dyn_clone::clone_box`] to clone the value if required.
    pub fn get_value<'a>(&self, ctx: &'a Context) -> Ref<'a, dyn TypedAttrInterface> {
        Ref::map(
            self.get_attr_onnx_constant_value(ctx)
                .expect("onnx.constant must have a value attribute"),
            |attr| {
                attr_cast::<dyn TypedAttrInterface>(&**attr)
                    .expect("onnx.constant value attribute must implement TypedAttrInterface")
            },
        )
    }

    /// Move the value attribute out of this constant.
    ///
    /// **WARNING**: `self` will fail to verify after this operation.
    pub fn take_value(&self, ctx: &Context) -> AttrObj {
        self.get_operation()
            .deref_mut(ctx)
            .attributes
            .0
            .remove(&constant_op_attr_names::ATTR_KEY_ONNX_CONSTANT_VALUE)
            .expect("onnx.constant must have a value attribute")
    }
}

impl Verify for ConstantOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let result_type = self.result_type(ctx);

        let Some(value) = self.get_attr_onnx_constant_value(ctx) else {
            return verify_err!(loc, ConstantOpVerifyErr::MissingValue);
        };
        let Some(typed_value) = attr_cast::<dyn TypedAttrInterface>(&**value) else {
            return verify_err!(
                loc,
                ConstantOpVerifyErr::ValueNotTyped(value.get_attr_id().to_string())
            );
        };

        if typed_value.get_type(ctx) != result_type {
            return verify_err!(
                loc,
                ConstantOpVerifyErr::ValueTypeMismatch {
                    result: result_type.disp(ctx).to_string(),
                    value: typed_value.get_type(ctx).disp(ctx).to_string(),
                }
            );
        }
        Ok(())
    }
}

/// Element-wise addition with implicit ONNX broadcasting, including host scalars.
///
/// ### Operand(s):
///
/// | operand | description |
/// |-----|-------|
/// | `lhs` | The left tensor or scalar. |
/// | `rhs` | The right tensor or scalar. |
///
/// ### Result(s):
///
/// | result | description |
/// |-----|-------|
/// | `result` | The sum. Its shape is the broadcast shape of the operands. |
#[pliron_op(
    name = "onnx.add",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        NOpdsInterface<2>,
        OneResultInterface,
        BroadcastableBinaryOpInterface,
    ],
    verifier = "succ"
)]
pub struct AddOp;

impl AddOp {
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result_type: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![lhs, rhs],
            vec![],
            0,
        );
        AddOp { op }
    }

    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

/// Element-wise multiplication with implicit ONNX broadcasting, including host scalars.
#[pliron_op(
    name = "onnx.mul",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        NOpdsInterface<2>,
        OneResultInterface,
        BroadcastableBinaryOpInterface,
    ],
    verifier = "succ",
)]
pub struct MulOp;

impl MulOp {
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result_type: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![lhs, rhs],
            vec![],
            0,
        );
        MulOp { op }
    }

    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

/// Shape-preserving element-type cast. The result type encodes ONNX `to`.
///
/// ### Operand(s):
///
/// | operand | description |
/// |-----|-------|
/// | `input` | The tensor or scalar to convert. |
///
/// ### Result(s):
///
/// | result | description |
/// |-----|-------|
/// | `result` | The converted value, with the input shape. |
#[pliron_op(
    name = "onnx.cast",
    format = "$0 ` to ` type($0) ` ` attr($onnx_cast_signedness, $CastSignednessAttr)",
    interfaces = [OneOpdInterface, NOpdsInterface<1>, OneResultInterface],
    attributes = (onnx_cast_signedness: CastSignednessAttr),
)]
pub struct CastOp;

#[derive(Debug, thiserror::Error)]
pub enum CastOpVerifyErr {
    #[error("{0} is not a valid operand or result type for onnx.cast")]
    UnsupportedType(String),
    #[error("onnx.cast must keep the shape, but it changes {from} to {to}")]
    ShapeMismatch { from: String, to: String },
}

impl CastOp {
    pub fn new(
        ctx: &mut Context,
        input: Value,
        result_type: TypeHandle,
        signedness: CastSignednessAttr,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![input],
            vec![],
            0,
        );
        let op = CastOp { op };
        op.set_attr_onnx_cast_signedness(ctx, signedness);
        op
    }

    pub fn signedness(&self, ctx: &Context) -> CastSignednessAttr {
        *self
            .get_attr_onnx_cast_signedness(ctx)
            .expect("onnx.cast must have a signedness attribute")
    }
}

impl Verify for CastOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let operand_type = self.get_operand(ctx).get_type(ctx);
        let result_type = self.result_type(ctx);

        let shape_of = |ty: TypeHandle| {
            value_shape(ctx, ty).ok_or_else(|| {
                verify_error!(
                    loc.clone(),
                    CastOpVerifyErr::UnsupportedType(ty.disp(ctx).to_string())
                )
            })
        };

        if shape_of(operand_type)? != shape_of(result_type)? {
            return verify_err!(
                loc,
                CastOpVerifyErr::ShapeMismatch {
                    from: operand_type.disp(ctx).to_string(),
                    to: result_type.disp(ctx).to_string(),
                }
            );
        }
        Ok(())
    }
}

/// Return graph outputs from an ONNX function.
///
/// ### Operand(s):
///
/// | operand | description |
/// |-----|-------|
/// | `outputs` | One operand for each graph output. |
#[pliron_op(
    name = "onnx.return",
    format = "operands(CharSpace(`,`))",
    interfaces = [IsTerminatorInterface, NResultsInterface<0>],
    verifier = "succ",
)]
pub struct ReturnOp;

impl ReturnOp {
    pub fn new(ctx: &mut Context, outputs: Vec<Value>) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            outputs,
            vec![],
            0,
        );
        ReturnOp { op }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BroadcastableBinaryOpVerifyErr {
    #[error("{0} is not a valid operand or result type for a binary broadcast operation")]
    UnsupportedType(String),
    #[error("The operand shapes {lhs} and {rhs} are not compatible")]
    IncompatibleShapes { lhs: String, rhs: String },
    #[error("The operands give the broadcast shape {expected}, but the result has the shape {got}")]
    ResultShapeMismatch { expected: String, got: String },
    #[error("The operand and result element types are different: {lhs}, {rhs} and {result}")]
    ElementTypeMismatch {
        lhs: String,
        rhs: String,
        result: String,
    },
}

/// Operations whose two operands use ONNX broadcasting and preserve their element type.
#[op_interface]
pub trait BroadcastableBinaryOpInterface: NOpdsInterface<2> + OneResultInterface {
    /// Check ONNX broadcast compatibility and matching operand/result element types.
    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let op = op_cast::<dyn BroadcastableBinaryOpInterface>(op)
            .expect("operation must implement BroadcastableBinaryOpInterface");
        let loc = op.loc(ctx);
        let (lhs_ty, rhs_ty) = {
            let op_ref = op.get_operation().deref(ctx);
            (
                op_ref.get_operand(0).get_type(ctx),
                op_ref.get_operand(1).get_type(ctx),
            )
        };
        let result_ty = op.result_type(ctx);

        let shape_of = |ty: TypeHandle| -> Result<Vec<Dimension>> {
            value_shape(ctx, ty).ok_or_else(|| {
                verify_error!(
                    loc.clone(),
                    BroadcastableBinaryOpVerifyErr::UnsupportedType(ty.disp(ctx).to_string())
                )
            })
        };
        let (lhs_shape, rhs_shape, result_shape) =
            (shape_of(lhs_ty)?, shape_of(rhs_ty)?, shape_of(result_ty)?);

        let broadcast = broadcast_shapes(&lhs_shape, &rhs_shape).ok_or_else(|| {
            verify_error!(
                loc.clone(),
                BroadcastableBinaryOpVerifyErr::IncompatibleShapes {
                    lhs: disp_shape(ctx, &lhs_shape),
                    rhs: disp_shape(ctx, &rhs_shape),
                }
            )
        })?;

        // Unknown extents cannot rule out shape compatibility.
        if broadcast.len() != result_shape.len()
            || broadcast.iter().zip(result_shape.iter()).any(
                |(b, r)| matches!((b, r), (Dimension::Static(b), Dimension::Static(r)) if b != r),
            )
        {
            return verify_err!(
                loc,
                BroadcastableBinaryOpVerifyErr::ResultShapeMismatch {
                    expected: disp_shape(ctx, &broadcast),
                    got: disp_shape(ctx, &result_shape),
                }
            );
        }

        let element_types = [lhs_ty, rhs_ty, result_ty].map(|ty| value_element_type(ctx, ty));
        if element_types[0] != element_types[1] || element_types[1] != element_types[2] {
            let disp = |ty: Option<TypeHandle>| {
                ty.map_or_else(|| "<none>".to_string(), |ty| ty.disp(ctx).to_string())
            };
            return verify_err!(
                loc,
                BroadcastableBinaryOpVerifyErr::ElementTypeMismatch {
                    lhs: disp(element_types[0]),
                    rhs: disp(element_types[1]),
                    result: disp(element_types[2]),
                }
            );
        }

        Ok(())
    }
}

fn disp_shape(ctx: &Context, shape: &[Dimension]) -> String {
    if shape.is_empty() {
        return "<scalar>".to_string();
    }
    shape
        .iter()
        .map(|dim| dim.disp(ctx).to_string())
        .collect::<Vec<_>>()
        .join("x")
}
