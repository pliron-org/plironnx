// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/identity/mod.rs (burn-onnx repo).

#[allow(dead_code)]
mod identity_only_model {
    include!(concat!(env!("OUT_DIR"), "/model/identity_only.rs"));
}

use burn::tensor::{Device, Tensor};

mod common;
#[path = "common/single_input_f32.rs"]
mod single_input_f32;
mod utils;

#[test]
fn identity_only_ir() {
    utils::assert_ir_snapshots(
        "identity/identity_only.onnx",
        [
            "identity_only_onnx.expect.plir",
            "identity_only_tensor.expect.plir",
            "identity_only_llvm.expect.plir",
        ],
    );
}

#[test]
fn identity_only() {
    let device: Device = Default::default();
    let input = Tensor::<2>::from_floats(
        [[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]],
        &device,
    );
    let expected = identity_only_model::Model::default().forward(input.clone());
    let actual =
        single_input_f32::run_plironnx("identity/identity_only.onnx", &input, input.dims());
    actual.to_data().assert_eq(&expected.to_data(), true);
}
