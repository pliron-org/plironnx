// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! ONNX constant data to dense tensor or builtin scalar attributes.

use onnx_ir::{DType, TensorData};

use pliron::{
    builtin::attr_interfaces::TypedAttrInterface, context::Context, input_err_noloc,
    input_error_noloc, printable::Printable, result::Result, r#type::TypeHandle,
};
use pliron_tensor::memref::{
    attributes::{DenseElementsAttr, ShapedTypeHandle},
    type_interfaces::DenseElementTypeHandle,
};

#[derive(Debug, thiserror::Error)]
pub enum ConstantDataErr {
    #[error("{0} is not a supported element type for a constant")]
    UnsupportedElementType(String),
    #[error("A constant of type {0} needs a fully static shape")]
    DynamicShape(String),
    #[error(
        "An ONNX constant of the element type {dtype} has {got} byte(s) of data, \
         but {ty} needs {expected}"
    )]
    DTypeSizeMismatch {
        dtype: String,
        ty: String,
        expected: usize,
        got: usize,
    },
    #[error("{0} cannot be the result type of a constant")]
    UnsupportedResultType(String),
}

/// Build a tensor or scalar constant attribute, compressing uniform data to a splat.
pub fn constant_value_attr(
    ctx: &Context,
    data: &TensorData,
    result_type: TypeHandle,
) -> Result<Box<dyn TypedAttrInterface>> {
    constant_value_attr_from_bytes(ctx, data.dtype(), data.as_bytes().to_vec(), result_type)
}

/// Build a tensor or scalar constant attribute
/// from owned `bytes` of the element type `dtype`,
/// compressing uniform data to a splat.
pub fn constant_value_attr_from_bytes(
    ctx: &Context,
    dtype: DType,
    bytes: Vec<u8>,
    result_type: TypeHandle,
) -> Result<Box<dyn TypedAttrInterface>> {
    if let Ok(shaped_ty) = ShapedTypeHandle::from_handle(result_type, ctx) {
        let (element_size, num_elements) = {
            let shaped = shaped_ty.deref(ctx);
            let element_type = shaped.element_type();
            let element_size = DenseElementTypeHandle::from_handle(element_type, ctx)
                .map_err(|_| {
                    input_error_noloc!(ConstantDataErr::UnsupportedElementType(
                        element_type.disp(ctx).to_string()
                    ))
                })?
                .deref(ctx)
                .element_size();
            let num_elements = shaped.num_elements().ok_or_else(|| {
                input_error_noloc!(ConstantDataErr::DynamicShape(
                    result_type.disp(ctx).to_string()
                ))
            })?;
            (element_size, num_elements)
        };

        let bytes = checked_bytes(ctx, dtype, bytes, element_size, num_elements, result_type)?;
        Ok(Box::new(DenseElementsAttr::new(ctx, shaped_ty, bytes)?))
    } else if let Ok(element_ty) = DenseElementTypeHandle::from_handle(result_type, ctx) {
        let element_size = element_ty.deref(ctx).element_size();
        let bytes = checked_bytes(ctx, dtype, bytes, element_size, 1, result_type)?;
        Ok(element_ty.deref(ctx).element_attr(ctx, &bytes))
    } else {
        input_err_noloc!(ConstantDataErr::UnsupportedResultType(
            result_type.disp(ctx).to_string()
        ))
    }
}

/// Validate constant data and compress uniform multi-element values to a splat.
fn checked_bytes(
    ctx: &Context,
    dtype: DType,
    mut bytes: Vec<u8>,
    element_size: usize,
    num_elements: usize,
    ty: TypeHandle,
) -> Result<Vec<u8>> {
    if dtype.size() != element_size || bytes.len() != num_elements * element_size {
        return input_err_noloc!(ConstantDataErr::DTypeSizeMismatch {
            dtype: format!("{dtype:?}"),
            ty: ty.disp(ctx).to_string(),
            expected: num_elements * element_size,
            got: bytes.len(),
        });
    }

    if num_elements > 1 {
        let first = &bytes[..element_size];
        if bytes.chunks(element_size).all(|chunk| chunk == first) {
            // This is a splat.
            bytes.truncate(element_size);
        }
    }
    Ok(bytes)
}
