// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/sub/mod.rs (burn-onnx repo).

#[allow(dead_code)]
mod sub_int_model {
    include!(concat!(env!("OUT_DIR"), "/model/sub_int.rs"));
}

use burn::tensor::{DType, Device, Int, Tensor, TensorData};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

#[test]
fn sub_int_ir() {
    utils::assert_ir_snapshots(
        "sub/sub_int.onnx",
        [
            "sub_int_onnx.expect.plir",
            "sub_int_tensor.expect.plir",
            "sub_int_llvm.expect.plir",
        ],
    );
}

/// Check that i64 subtraction of a scalar and of a tensor matches Burn.
#[test]
fn sub_scalar_from_int_tensor_and_int_tensor_from_tensor() {
    let device: Device = Default::default();
    let input = Tensor::<4, Int>::from_data(
        TensorData::from([[[[1i64, 2, 3, 4]]]]),
        (&device, DType::I64),
    );
    let scalar = 3i64;
    let expected = sub_int_model::Model::default().forward(input.clone(), scalar);

    let input_data = input.to_data();
    let input_descriptor = jit::build_input_descriptor::<i64>(&input_data);
    let mut output = jit::build_empty_output_descriptor(4);
    jit::run_packed(
        "sub/sub_int.onnx",
        &[input_descriptor.as_ptr(), (&raw const scalar).cast()],
        &[output.as_mut_ptr()],
    );
    assert_eq!(
        jit::read_output::<i64>(&output, &[1, 1, 1, 4]),
        expected.to_data().as_slice::<i64>().unwrap()
    );
}
