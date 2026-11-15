// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/flatten/mod.rs (burn-onnx repo).

#[allow(dead_code)]
mod flatten_model {
    include!(concat!(env!("OUT_DIR"), "/model/flatten.rs"));
}
#[allow(dead_code)]
mod flatten_2d_model {
    include!(concat!(env!("OUT_DIR"), "/model/flatten_2d.rs"));
}
#[allow(dead_code)]
mod flatten_rank1_model {
    include!(concat!(env!("OUT_DIR"), "/model/flatten_rank1.rs"));
}

use burn::tensor::{Device, Int, Tensor};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

#[test]
fn flatten_ir() {
    utils::assert_ir_snapshots(
        "flatten/flatten_2d.onnx",
        [
            "flatten_onnx.expect.plir",
            "flatten_tensor.expect.plir",
            "flatten_llvm.expect.plir",
        ],
    );
}

/// A tensor of the given shape whose elements are 0, 1, 2, ... in row-major order.
fn iota<const D: usize>(shape: [usize; D]) -> Tensor<D> {
    let len = shape.iter().product::<usize>() as i64;
    Tensor::<1, Int>::arange(0..len, &Default::default())
        .float()
        .reshape(shape)
}

/// Check that the flattened tensors keep their elements in row-major order, as in Burn:
/// - axis 1 of a rank-3 tensor
/// - axis 2 of a rank-4 tensor
/// - axis 0 of a rank-1 tensor, which adds a leading unit dimension
#[test]
fn flatten() {
    let device: Device = Default::default();

    let input = iota([1, 5, 15]);
    let expected = flatten_model::Model::new(&device).forward(input.clone());
    let actual = jit::run_single_f32("flatten/flatten.onnx", &input, [1, 75]);
    actual.to_data().assert_eq(&expected.to_data(), true);

    let input = iota([2, 3, 4, 5]);
    let expected = flatten_2d_model::Model::new(&device).forward(input.clone());
    let actual = jit::run_single_f32("flatten/flatten_2d.onnx", &input, [6, 20]);
    actual.to_data().assert_eq(&expected.to_data(), true);

    let input = iota([5]);
    let expected = flatten_rank1_model::Model::new(&device).forward(input.clone());
    let actual = jit::run_single_f32("flatten/flatten_rank1.onnx", &input, [1, 5]);
    actual.to_data().assert_eq(&expected.to_data(), true);
}
