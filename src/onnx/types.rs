// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Maps ONNX argument types to pliron ranked tensors and builtin scalars,
//! and provides their common shape and element-type queries

use onnx_ir::{ArgType, DType, TensorType};

use pliron::{
    arg_err_noloc,
    builtin::{
        type_interfaces::FloatTypeInterface,
        types::{FP16Type, FP32Type, FP64Type, IntegerType, Signedness},
    },
    context::Context,
    result::Result,
    r#type::{Type, TypeHandle, TypedHandle, type_impls},
};
use pliron_tensor::{
    memref::type_interfaces::{Dimension, MultiDimensionalType, ShapedType},
    tensor::types::RankedTensorType,
};

#[derive(Debug, thiserror::Error)]
pub enum TypeConversionErr {
    #[error("ONNX element type {0} is not supported")]
    UnsupportedDType(String),
    #[error("The ONNX argument type Shape({0}) is not supported")]
    UnsupportedShapeArg(usize),
}

/// Converts an ONNX element type to its pliron equivalent.
pub fn dtype_to_type(ctx: &Context, dtype: DType) -> Result<TypeHandle> {
    let ty: TypeHandle = match dtype {
        DType::F16 => FP16Type::get(ctx).into(),
        DType::F32 => FP32Type::get(ctx).into(),
        DType::F64 => FP64Type::get(ctx).into(),
        DType::I8 | DType::U8 => IntegerType::get(ctx, 8, Signedness::Signless).into(),
        DType::I16 | DType::U16 => IntegerType::get(ctx, 16, Signedness::Signless).into(),
        DType::I32 | DType::U32 => IntegerType::get(ctx, 32, Signedness::Signless).into(),
        DType::I64 | DType::U64 => IntegerType::get(ctx, 64, Signedness::Signless).into(),
        DType::Bool(_) => IntegerType::get(ctx, 1, Signedness::Signless).into(),
        other => {
            return arg_err_noloc!(TypeConversionErr::UnsupportedDType(format!("{other:?}")));
        }
    };
    Ok(ty)
}

/// Returns the element type carried by an ONNX argument.
pub fn arg_dtype(ty: &ArgType) -> Option<DType> {
    match ty {
        ArgType::ScalarNative(dtype) | ArgType::ScalarTensor(dtype) => Some(*dtype),
        ArgType::Tensor(tensor_ty) => Some(tensor_ty.dtype),
        ArgType::Shape(_) => None,
    }
}

/// Returns whether an ONNX argument has a signed integer element type.
pub fn is_arg_element_signed(ty: &ArgType) -> bool {
    matches!(
        arg_dtype(ty),
        Some(DType::I8 | DType::I16 | DType::I32 | DType::I64)
    )
}

/// Converts an ONNX tensor shape to pliron dimensions.
pub fn tensor_shape(ty: &TensorType) -> Vec<Dimension> {
    match &ty.static_shape {
        Some(dims) => dims
            .iter()
            .map(|dim| dim.map_or(Dimension::Dynamic, Dimension::Static))
            .collect(),
        None => vec![Dimension::Dynamic; ty.rank],
    }
}

/// Converts an ONNX tensor type to a ranked tensor type.
pub fn tensor_type_to_type(
    ctx: &Context,
    ty: &TensorType,
) -> Result<TypedHandle<RankedTensorType>> {
    let element_type = dtype_to_type(ctx, ty.dtype)?;
    Ok(RankedTensorType::get(ctx, element_type, tensor_shape(ty)))
}

/// Converts an ONNX argument type to its pliron equivalent.
pub fn arg_type_to_type(ctx: &Context, ty: &ArgType) -> Result<TypeHandle> {
    match ty {
        ArgType::ScalarNative(dtype) => dtype_to_type(ctx, *dtype),
        // burn represents device scalars as rank-1 tensors of length 1.
        ArgType::ScalarTensor(dtype) => {
            let element_type = dtype_to_type(ctx, *dtype)?;
            Ok(RankedTensorType::get(ctx, element_type, vec![Dimension::Static(1)]).into())
        }
        ArgType::Tensor(tensor_ty) => Ok(tensor_type_to_type(ctx, tensor_ty)?.into()),
        ArgType::Shape(rank) => arg_err_noloc!(TypeConversionErr::UnsupportedShapeArg(*rank)),
    }
}

/// Returns whether a type is an integer or floating-point scalar.
pub fn is_scalar_type(ty: &dyn Type) -> bool {
    ty.is::<IntegerType>() || type_impls::<dyn FloatTypeInterface>(ty)
}

/// Returns a value's broadcast shape, or `None` for unsupported types.
pub fn value_shape(ctx: &Context, ty: TypeHandle) -> Option<Vec<Dimension>> {
    let ty_ref = &*ty.deref(ctx);
    if let Some(tensor_ty) = ty_ref.downcast_ref::<RankedTensorType>() {
        Some(tensor_ty.shape().clone())
    } else if is_scalar_type(ty_ref) {
        Some(vec![])
    } else {
        None
    }
}

/// Returns a value's element type, or `None` for unsupported types.
pub fn value_element_type(ctx: &Context, ty: TypeHandle) -> Option<TypeHandle> {
    let ty_ref = &*ty.deref(ctx);
    if let Some(tensor_ty) = ty_ref.downcast_ref::<RankedTensorType>() {
        Some(tensor_ty.element_type())
    } else if is_scalar_type(ty_ref) {
        Some(ty)
    } else {
        None
    }
}
