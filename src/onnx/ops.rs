// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! ONNX dialect operations.

use core::cell::Ref;
use pliron::{
    attribute::{AttrObj, attr_cast},
    builtin::{
        attr_interfaces::TypedAttrInterface,
        attributes::BoolAttr,
        op_interfaces::{
            IsTerminatorInterface, NOpdsInterface, NResultsInterface, OneOpdInterface,
            OneResultInterface, OperandNOfType, ResultNImplsTy, ResultNOfType,
        },
        type_interfaces::FloatTypeInterface,
    },
    common_traits::Verify,
    context::Context,
    derive::{op_interface, op_interface_impl, pliron_op},
    dict_key,
    op::{Op, op_cast},
    operation::Operation,
    printable::Printable,
    result::Result,
    r#type::{TypeHandle, Typed, TypedHandle, type_impls},
    value::Value,
    verify_err, verify_error,
};
use pliron_tensor::{
    memref::{
        attributes::{PermutationAttr, ReductionDimsAttr},
        layout::permute,
        op_interfaces::{
            DynamicDimensionOperandsOp, ReshapeOpInterface, TransposeOpInterface, reduced_shape,
        },
        type_interfaces::{Dimension, MultiDimensionalType, ShapedType},
    },
    tensor::{
        attributes::CastSignednessAttr,
        types::{RankedTensorType, broadcast_shapes},
    },
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
        let op = Operation::new_op::<Self>(ctx, vec![result_type], vec![], vec![], 0);
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
        Operation::new_op::<Self>(ctx, vec![result_type], vec![lhs, rhs], vec![], 0)
    }
}

/// Element-wise subtraction with implicit ONNX broadcasting, including host scalars.
#[pliron_op(
    name = "onnx.sub",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        NOpdsInterface<2>,
        OneResultInterface,
        BroadcastableBinaryOpInterface,
    ],
    verifier = "succ",
)]
pub struct SubOp;

impl SubOp {
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result_type: TypeHandle) -> Self {
        Operation::new_op::<Self>(ctx, vec![result_type], vec![lhs, rhs], vec![], 0)
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
        Operation::new_op::<Self>(ctx, vec![result_type], vec![lhs, rhs], vec![], 0)
    }
}

/// Element-wise division with implicit ONNX broadcasting, including host scalars.
/// Integer division truncates toward zero.
#[pliron_op(
    name = "onnx.div",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        NOpdsInterface<2>,
        OneResultInterface,
        BroadcastableBinaryOpInterface,
    ],
    verifier = "succ",
)]
pub struct DivOp;

impl DivOp {
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result_type: TypeHandle) -> Self {
        Operation::new_op::<Self>(ctx, vec![result_type], vec![lhs, rhs], vec![], 0)
    }
}

/// Rectified linear unit. Elementwise `max(x, 0)`.
///
/// ### Operand(s):
///
/// | operand | description |
/// |-----|-------|
/// | `input` | The tensor or scalar, of a float or signed integer type. |
///
/// ### Result(s):
///
/// | result | description |
/// |-----|-------|
/// | `result` | The rectified value, with the type of `input`. |
#[pliron_op(
    name = "onnx.relu",
    format = "$0 ` : ` type($0)",
    interfaces = [OneOpdInterface, NOpdsInterface<1>, OneResultInterface],
)]
pub struct ReluOp;

#[derive(Debug, thiserror::Error)]
pub enum ReluOpVerifyErr {
    #[error(
        "The operand of onnx.relu has the type {operand}, but the result has the type {result}"
    )]
    TypeMismatch { operand: String, result: String },
    #[error("{0} is not a valid operand type for onnx.relu")]
    UnsupportedType(String),
}

impl ReluOp {
    /// Create a new ReluOp. The result type is set to `input`'s type.
    pub fn new(ctx: &mut Context, input: Value) -> Self {
        let result_type = input.get_type(ctx);
        Operation::new_op::<Self>(ctx, vec![result_type], vec![input], vec![], 0)
    }
}

impl Verify for ReluOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let operand_type = self.get_operand(ctx).get_type(ctx);
        let result_type = self.result_type(ctx);
        if operand_type != result_type {
            return verify_err!(
                loc,
                ReluOpVerifyErr::TypeMismatch {
                    operand: operand_type.disp(ctx).to_string(),
                    result: result_type.disp(ctx).to_string(),
                }
            );
        }
        if value_element_type(ctx, operand_type).is_none() {
            return verify_err!(
                loc,
                ReluOpVerifyErr::UnsupportedType(operand_type.disp(ctx).to_string())
            );
        }
        Ok(())
    }
}

/// Elementwise base-e exponential, as defined by ONNX `Exp`.
///
/// ### Operand(s):
///
/// | operand | description |
/// |-----|-------|
/// | `input` | The tensor or scalar, of a float type. |
///
/// ### Result(s):
///
/// | result | description |
/// |-----|-------|
/// | `result` | The exponential, with the type of `input`. |
#[pliron_op(
    name = "onnx.exp",
    format = "$0 ` : ` type($0)",
    interfaces = [OneOpdInterface, NOpdsInterface<1>, OneResultInterface],
)]
pub struct ExpOp;

#[derive(Debug, thiserror::Error)]
pub enum ExpOpVerifyErr {
    #[error("The operand of onnx.exp has the type {operand}, but the result has the type {result}")]
    TypeMismatch { operand: String, result: String },
    #[error("{0} is not a valid operand type for onnx.exp")]
    UnsupportedType(String),
}

impl ExpOp {
    /// Create a new ExpOp. The result type is set to `input`'s type.
    pub fn new(ctx: &mut Context, input: Value) -> Self {
        let result_type = input.get_type(ctx);
        Operation::new_op::<Self>(ctx, vec![result_type], vec![input], vec![], 0)
    }
}

impl Verify for ExpOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let operand_type = self.get_operand(ctx).get_type(ctx);
        let result_type = self.result_type(ctx);
        if operand_type != result_type {
            return verify_err!(
                loc,
                ExpOpVerifyErr::TypeMismatch {
                    operand: operand_type.disp(ctx).to_string(),
                    result: result_type.disp(ctx).to_string(),
                }
            );
        }
        let is_float = value_element_type(ctx, operand_type)
            .is_some_and(|ty| type_impls::<dyn FloatTypeInterface>(&*ty.deref(ctx)));
        if !is_float {
            return verify_err!(
                loc,
                ExpOpVerifyErr::UnsupportedType(operand_type.disp(ctx).to_string())
            );
        }
        Ok(())
    }
}

dict_key!(
    /// Attribute key for the [ReductionDimsAttr] of a [ReduceOpInterface] op.
    ATTR_KEY_REDUCE_DIMS,
    "onnx_reduce_dims"
);

dict_key!(
    /// Attribute key for the `keepdims` [BoolAttr] of a [ReduceOpInterface] op.
    ATTR_KEY_REDUCE_KEEPDIMS,
    "onnx_reduce_keepdims"
);

#[derive(Debug, thiserror::Error)]
pub enum ReduceOpVerifyErr {
    #[error("The reduction has no {ATTR_KEY_REDUCE_DIMS} attribute")]
    MissingDims,
    #[error("The reduction has no {ATTR_KEY_REDUCE_KEEPDIMS} attribute")]
    MissingKeepdims,
    #[error("The dims {dims:?} must be strictly increasing dimensions of the rank {rank} input")]
    InvalidDims { dims: Vec<usize>, rank: usize },
    #[error("The input and the result element types are different: {input} and {result}")]
    ElementTypeMismatch { input: String, result: String },
    #[error("The reduction gives the result shape {expected}, but the result has the shape {got}")]
    ResultShapeMismatch { expected: String, got: String },
}

/// Compute the result shape when `input_shape` is reduced over `reduced_dims`.
///
/// Each dimension of `input_shape` that is in `reduced_dims` is set to 1 if `keepdims`
/// is set, and is removed otherwise.
pub fn onnx_reduced_shape(
    input_shape: &[Dimension],
    reduced_dims: &[usize],
    keepdims: bool,
) -> Vec<Dimension> {
    if keepdims {
        return reduced_shape(input_shape, reduced_dims);
    }
    input_shape
        .iter()
        .enumerate()
        .filter(|(dim, _)| !reduced_dims.contains(dim))
        .map(|(_, extent)| extent.clone())
        .collect()
}

/// ONNX reductions of a tensor over `reduced_dims`.
///
/// ONNX's convention for an empty `axes` is to reduce along all dimensions.
/// Because our convention is to not reduce at all on an empty `reduced_dims`,
/// the import translates an empty `axes` into a full `reduced_dims`.
#[op_interface]
pub trait ReduceOpInterface:
    OneOpdInterface
    + OneResultInterface
    + OperandNOfType<0, RankedTensorType>
    + ResultNOfType<0, RankedTensorType>
{
    /// Reduced dimensions, in strictly increasing order.
    fn reduced_dims(&self, ctx: &Context) -> Vec<usize> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<ReductionDimsAttr>(&ATTR_KEY_REDUCE_DIMS)
            .expect("the verifier checks that a reduction has reduced dims")
            .0
            .clone()
    }

    /// Are reduced dimensions retained with a unit extent or removed entirely?
    fn keepdims(&self, ctx: &Context) -> bool {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<BoolAttr>(&ATTR_KEY_REDUCE_KEEPDIMS)
            .expect("the verifier checks that a reduction has keepdims")
            .clone()
            .into()
    }

    /// Set reduced dimensions and `keepdims`.
    fn set_reduction(&self, ctx: &Context, reduced_dims: Vec<usize>, keepdims: bool) {
        let attributes = &mut self.get_operation().deref_mut(ctx).attributes;
        attributes.set(
            ATTR_KEY_REDUCE_DIMS.clone(),
            ReductionDimsAttr(reduced_dims),
        );
        attributes.set(ATTR_KEY_REDUCE_KEEPDIMS.clone(), BoolAttr::new(keepdims));
    }

    // Check reduced dims, element types and the result shape.
    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let loc = op.loc(ctx);
        let op_ref = op.get_operation().deref(ctx);
        let Some(ReductionDimsAttr(reduced_dims)) = op_ref
            .attributes
            .get::<ReductionDimsAttr>(&ATTR_KEY_REDUCE_DIMS)
        else {
            return verify_err!(loc, ReduceOpVerifyErr::MissingDims);
        };
        let Some(keepdims) = op_ref.attributes.get::<BoolAttr>(&ATTR_KEY_REDUCE_KEEPDIMS) else {
            return verify_err!(loc, ReduceOpVerifyErr::MissingKeepdims);
        };
        let keepdims: bool = keepdims.clone().into();

        let tensor_type = |value: Value| {
            TypedHandle::<RankedTensorType>::from_handle(value.get_type(ctx), ctx)
                .expect("the interfaces verify that the operand and the result are ranked tensors")
        };
        let input_ty = tensor_type(op_ref.get_operand(0));
        let result_ty = tensor_type(op_ref.get_result(0));
        let (input_ty, result_ty) = (input_ty.deref(ctx), result_ty.deref(ctx));

        let rank = input_ty.rank();
        let increasing = reduced_dims.windows(2).all(|pair| pair[0] < pair[1]);
        if !increasing || reduced_dims.last().is_some_and(|&last| last >= rank) {
            return verify_err!(
                loc,
                ReduceOpVerifyErr::InvalidDims {
                    dims: reduced_dims.clone(),
                    rank
                }
            );
        }

        if input_ty.element_type() != result_ty.element_type() {
            return verify_err!(
                loc,
                ReduceOpVerifyErr::ElementTypeMismatch {
                    input: input_ty.element_type().disp(ctx).to_string(),
                    result: result_ty.element_type().disp(ctx).to_string(),
                }
            );
        }

        let expected = onnx_reduced_shape(input_ty.shape(), reduced_dims, keepdims);
        if !shapes_may_match(&expected, result_ty.shape()) {
            return verify_err!(
                loc,
                ReduceOpVerifyErr::ResultShapeMismatch {
                    expected: disp_shape(ctx, &expected),
                    got: disp_shape(ctx, result_ty.shape()),
                }
            );
        }
        Ok(())
    }
}

/// Maximum of the elements over `reduced_dims`, as defined by ONNX `ReduceMax`.
///
/// - Unsigned integers compare as unsigned values.
/// - Signed and signless integers compare as signed values.
/// - Floats follow IEEE 754-2019 `maximum`:
///   - a NaN gives NaN
///   - `-0.0` is less than `+0.0`.
///
/// ### Operand(s):
///
/// | operand | description |
/// |-----|-------|
/// | `input` | The tensor to reduce. |
///
/// ### Result(s):
///
/// | result | description |
/// |-----|-------|
/// | `result` | The maximum. See [ReduceOpInterface] for its shape. |
#[pliron_op(
    name = "onnx.reduce_max",
    format = "$0 ` ` attr($onnx_reduce_dims, $ReductionDimsAttr) ` keepdims=` attr($onnx_reduce_keepdims, $BoolAttr) ` : ` type($0)",
    interfaces = [
        OneOpdInterface,
        OneResultInterface,
        OperandNOfType<0, RankedTensorType>,
        ResultNOfType<0, RankedTensorType>,
        ReduceOpInterface,
    ],
    verifier = "succ",
)]
pub struct ReduceMaxOp;

impl ReduceMaxOp {
    /// Create a new ReduceMaxOp: the maximum of `input` over `reduced_dims`.
    pub fn new(
        ctx: &mut Context,
        input: Value,
        reduced_dims: Vec<usize>,
        keepdims: bool,
        result_type: TypedHandle<RankedTensorType>,
    ) -> Self {
        let op = Operation::new_op::<Self>(ctx, vec![result_type.into()], vec![input], vec![], 0);
        op.set_reduction(ctx, reduced_dims, keepdims);
        op
    }
}

/// Sum of the elements over `reduced_dims`, as defined by ONNX `ReduceSum`.
///
/// Integer sums wrap around.
///
/// ### Operand(s):
///
/// | operand | description |
/// |-----|-------|
/// | `input` | The tensor to reduce. |
///
/// ### Result(s):
///
/// | result | description |
/// |-----|-------|
/// | `result` | The sum. See [ReduceOpInterface] for its shape. |
#[pliron_op(
    name = "onnx.reduce_sum",
    format = "$0 ` ` attr($onnx_reduce_dims, $ReductionDimsAttr) ` keepdims=` attr($onnx_reduce_keepdims, $BoolAttr) ` : ` type($0)",
    interfaces = [
        OneOpdInterface,
        OneResultInterface,
        OperandNOfType<0, RankedTensorType>,
        ResultNOfType<0, RankedTensorType>,
        ReduceOpInterface,
    ],
    verifier = "succ",
)]
pub struct ReduceSumOp;

impl ReduceSumOp {
    /// Create a new ReduceSumOp: the sum of `input` over `reduced_dims`.
    pub fn new(
        ctx: &mut Context,
        input: Value,
        reduced_dims: Vec<usize>,
        keepdims: bool,
        result_type: TypedHandle<RankedTensorType>,
    ) -> Self {
        let op = Operation::new_op::<Self>(ctx, vec![result_type.into()], vec![input], vec![], 0);
        op.set_reduction(ctx, reduced_dims, keepdims);
        op
    }
}

/// Matrix product with NumPy semantics, as defined by ONNX `MatMul`.
///
/// | `lhs`                 | `rhs`                 | `result`      |
/// |-----------------------|-----------------------|---------------|
/// | `[..., M, K]`         | `[..., K, N]`         | `[..., M, N]` |
/// | `[K]` (as `[1, K]`)   | `[..., K, N]`         | `[..., N]`    |
/// | `[..., M, K]`         | `[K]` (as `[K, 1]`)   | `[..., M]`    |
/// | `[K]` (as `[1, K]`)   | `[K]` (as `[K, 1]`)   | `[]`          |
///
/// The batch dimensions (`...`) broadcast.
///
/// ### Operand(s):
///
/// | operand | description |
/// |-----|-------|
/// | `lhs` | The left tensor, of shape `[..., M, K]` or `[K]`. |
/// | `rhs` | The right tensor, of shape `[..., K, N]` or `[K]`. |
///
/// ### Result(s):
///
/// | result | description |
/// |-----|-------|
/// | `result` | The product. |
#[pliron_op(
    name = "onnx.matmul",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        NOpdsInterface<2>,
        OneResultInterface,
        OperandNOfType<0, RankedTensorType>,
        OperandNOfType<1, RankedTensorType>,
        ResultNOfType<0, RankedTensorType>,
    ],
)]
pub struct MatMulOp;

#[derive(Debug, thiserror::Error)]
pub enum MatMulOpVerifyErr {
    #[error("The operands of onnx.matmul must have rank 1 or more")]
    ScalarOperand,
    #[error("onnx.matmul cannot multiply the shapes {lhs} and {rhs}")]
    IncompatibleShapes { lhs: String, rhs: String },
    #[error(
        "The operands of onnx.matmul give the result shape {expected}, but the result has the shape {got}"
    )]
    ResultShapeMismatch { expected: String, got: String },
    #[error(
        "The operand and result element types of onnx.matmul are different: {lhs}, {rhs} and {result}"
    )]
    ElementTypeMismatch {
        lhs: String,
        rhs: String,
        result: String,
    },
}

impl MatMulOp {
    pub fn new(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        result_type: TypedHandle<RankedTensorType>,
    ) -> Self {
        Operation::new_op::<Self>(ctx, vec![result_type.into()], vec![lhs, rhs], vec![], 0)
    }

    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

impl Verify for MatMulOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let tensor_type = |value: Value| {
            TypedHandle::<RankedTensorType>::from_handle(value.get_type(ctx), ctx)
                .expect("the interfaces verify that the operands and the result are ranked tensors")
        };
        let lhs_ty = tensor_type(self.lhs(ctx));
        let rhs_ty = tensor_type(self.rhs(ctx));
        let result_ty = tensor_type(self.get_result(ctx));
        let (lhs_ty, rhs_ty, result_ty) =
            (lhs_ty.deref(ctx), rhs_ty.deref(ctx), result_ty.deref(ctx));

        if lhs_ty.rank() == 0 || rhs_ty.rank() == 0 {
            return verify_err!(loc, MatMulOpVerifyErr::ScalarOperand);
        }
        let expected = matmul_shapes(lhs_ty.shape(), rhs_ty.shape())
            .ok_or_else(|| {
                verify_error!(
                    loc.clone(),
                    MatMulOpVerifyErr::IncompatibleShapes {
                        lhs: disp_shape(ctx, lhs_ty.shape()),
                        rhs: disp_shape(ctx, rhs_ty.shape()),
                    }
                )
            })?
            .result;
        if !shapes_may_match(&expected, result_ty.shape()) {
            return verify_err!(
                loc,
                MatMulOpVerifyErr::ResultShapeMismatch {
                    expected: disp_shape(ctx, &expected),
                    got: disp_shape(ctx, result_ty.shape()),
                }
            );
        }

        let element_type = lhs_ty.element_type();
        if rhs_ty.element_type() != element_type || result_ty.element_type() != element_type {
            return verify_err!(
                loc,
                MatMulOpVerifyErr::ElementTypeMismatch {
                    lhs: element_type.disp(ctx).to_string(),
                    rhs: rhs_ty.element_type().disp(ctx).to_string(),
                    result: result_ty.element_type().disp(ctx).to_string(),
                }
            );
        }
        Ok(())
    }
}

/// Shapes of an ONNX matrix product.
///
/// The shapes are taken after:
/// - vector operands are promoted to matrices
/// - batch dimensions are broadcast
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatMulShapes {
    /// The `lhs` shape after the promotion of a vector to a `[1, K]` matrix.
    pub lhs_matrix: Vec<Dimension>,
    /// The `rhs` shape after the promotion of a vector to a `[K, 1]` matrix.
    pub rhs_matrix: Vec<Dimension>,
    /// The `lhs` matrix shape with the broadcast batch dimensions.
    pub lhs_batched: Vec<Dimension>,
    /// The `rhs` matrix shape with the broadcast batch dimensions.
    pub rhs_batched: Vec<Dimension>,
    /// The shape of the matrix product of `lhs_batched` and `rhs_batched`, `[..., M, N]`.
    /// [`Self::result`] removes `M` if `lhs` is a vector, and `N` if `rhs` is a vector.
    pub result_batched: Vec<Dimension>,
    /// The ONNX result shape.
    pub result: Vec<Dimension>,
}

/// Compute [MatMulShapes] of `lhs x rhs` for compatible shapes.
pub fn matmul_shapes(lhs: &[Dimension], rhs: &[Dimension]) -> Option<MatMulShapes> {
    if lhs.is_empty() || rhs.is_empty() {
        // Rank-0 operands are incompatible with anything.
        return None;
    }
    // Promote a vector `lhs` to a `[1, K]` matrix and a vector `rhs` to a `[K, 1]` matrix.
    let lhs_matrix = if lhs.len() == 1 {
        vec![Dimension::Static(1), lhs[0].clone()]
    } else {
        lhs.to_vec()
    };
    let rhs_matrix = if rhs.len() == 1 {
        vec![rhs[0].clone(), Dimension::Static(1)]
    } else {
        rhs.to_vec()
    };
    let (lhs_batch, lhs_mk) = lhs_matrix.split_at(lhs_matrix.len() - 2);
    let (rhs_batch, rhs_kn) = rhs_matrix.split_at(rhs_matrix.len() - 2);
    let (m, lhs_k) = (lhs_mk[0].clone(), lhs_mk[1].clone());
    let (rhs_k, n) = (rhs_kn[0].clone(), rhs_kn[1].clone());
    // The two `K` extents must not conflict.
    if matches!((&lhs_k, &rhs_k), (Dimension::Static(l), Dimension::Static(r)) if l != r) {
        return None;
    }
    let batch = broadcast_shapes(lhs_batch, rhs_batch)?;
    let batched = |rows: &Dimension, cols: &Dimension| {
        let mut shape = batch.clone();
        shape.extend([rows.clone(), cols.clone()]);
        shape
    };

    // Add `M` only if `lhs` is not a vector, and `N` only if `rhs` is not a vector.
    let mut result = batch.clone();
    if lhs.len() != 1 {
        result.push(m.clone());
    }
    if rhs.len() != 1 {
        result.push(n.clone());
    }
    Some(MatMulShapes {
        lhs_batched: batched(&m, &lhs_k),
        rhs_batched: batched(&rhs_k, &n),
        result_batched: batched(&m, &n),
        lhs_matrix,
        rhs_matrix,
        result,
    })
}

/// Do `lhs` and `rhs` have the same rank without conflicting static extents?
fn shapes_may_match(lhs: &[Dimension], rhs: &[Dimension]) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs)
            .all(|(l, r)| !matches!((l, r), (Dimension::Static(l), Dimension::Static(r)) if l != r))
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
        let op = Operation::new_op::<Self>(ctx, vec![result_type], vec![input], vec![], 0);
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

/// Reshape a tensor using a compile-time ONNX shape. The inferred result type
/// contains the resolved dimensions, including any ONNX `-1` dimension.
///
/// ### Operand(s):
///
/// | operand | description |
/// |-----|-------|
/// | `input` | The tensor to reshape. |
///
/// ### Result(s):
///
/// | result | description |
/// |-----|-------|
/// | `result` | The input tensor's elements arranged in the resolved shape. |
#[pliron_op(
    name = "onnx.reshape",
    format = "$0 ` to ` type($0)",
    interfaces = [
        OneOpdInterface,
        OneResultInterface,
        OperandNOfType<0, RankedTensorType>,
        ResultNOfType<0, RankedTensorType>,
        ResultNImplsTy<0, dyn ShapedType>,
        ReshapeOpInterface<RankedTensorType>,
    ],
    verifier = "succ",
)]
pub struct ReshapeOp;

#[op_interface_impl]
impl DynamicDimensionOperandsOp for ReshapeOp {
    fn get_dynamic_dimensions(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().skip(1).collect()
    }
}

impl ReshapeOp {
    pub fn new(
        ctx: &mut Context,
        input: Value,
        result_type: TypedHandle<RankedTensorType>,
    ) -> Self {
        Operation::new_op::<Self>(ctx, vec![result_type.into()], vec![input], vec![], 0)
    }
}

/// [Permute](permute) the dimensions of a tensor by the given
/// [permutation](PermutationAttr), so that for each source index `j`,
/// `result[permute(j, permutation)] = source[j]`.
///
/// ### Operand(s):
///
/// | operand | description |
/// |-----|-------|
/// | `source` | The source ranked tensor. |
///
/// ### Result(s):
///
/// | result | description |
/// |-----|-------|
/// | `result` | A ranked tensor with the permuted shape of `source`. |
#[pliron_op(
    name = "onnx.transpose",
    format = "$0 ` ` attr($memref_transpose_permutation, $PermutationAttr) ` : ` type($0)",
    interfaces = [
        OneOpdInterface,
        OneResultInterface,
        OperandNOfType<0, RankedTensorType>,
        ResultNOfType<0, RankedTensorType>,
        TransposeOpInterface<RankedTensorType>,
    ],
    verifier = "succ",
)]
pub struct TransposeOp;

impl TransposeOp {
    /// Create a new `TransposeOp` with the result type inferred.
    pub fn new(ctx: &mut Context, source: Value, permutation: Vec<usize>) -> Self {
        let (element_type, shape) = {
            let source_ty = source.get_type(ctx);
            let source_ty = source_ty.deref(ctx);
            let source_ty = source_ty
                .downcast_ref::<RankedTensorType>()
                .expect("onnx.transpose source must be ranked");
            (
                source_ty.element_type(),
                permute(source_ty.shape(), &permutation),
            )
        };
        let result_type = RankedTensorType::get(ctx, element_type, shape);
        let op = Operation::new_op::<Self>(ctx, vec![result_type.into()], vec![source], vec![], 0);
        op.set_permutation(ctx, permutation);
        op
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
        Operation::new_op::<Self>(ctx, vec![], outputs, vec![], 0)
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
    /// The left operand.
    fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// The right operand.
    fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

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

        if !shapes_may_match(&broadcast, &result_shape) {
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
