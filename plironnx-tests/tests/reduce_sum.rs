// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/reduce/mod.rs (burn-onnx repo).

#[allow(dead_code)]
mod reduce_sum_model {
    include!(concat!(env!("OUT_DIR"), "/model/reduce_sum.rs"));
}

use burn::tensor::{Device, Tensor, TensorData, Tolerance};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

#[test]
fn reduce_sum_ir() {
    utils::assert_ir_snapshots(
        "reduce/reduce_sum.onnx",
        [
            "reduce_sum_onnx.expect.plir",
            "reduce_sum_tensor.expect.plir",
            "reduce_sum_llvm.expect.plir",
        ],
    );
}

/// Check that the six reductions of the model match Burn:
/// - all dimensions, without `keepdims` and with no axes and empty axes
/// - one dimension, with `keepdims` and a positive and a negative axis
/// - one and two dimensions, without `keepdims`.
///
/// The axes of `ReduceSum` come from a constant input.
#[test]
fn reduce_sum() {
    let device: Device = Default::default();
    let input = Tensor::<4>::from_floats(
        [[[[1.5, 4.0, -9.0, 25.0], [-2.0, -5.0, -10.0, -26.0]]]],
        &device,
    );
    let (all, all_empty_axes, axis1, last_axis, axis0, axes02) =
        reduce_sum_model::Model::default().forward(input.clone());
    let actual = jit::run_f32_outputs(
        "reduce/reduce_sum.onnx",
        &input,
        &[
            &[1],
            &[1],
            &[1, 1, 2, 4],
            &[1, 1, 2, 1],
            &[1, 2, 4],
            &[1, 4],
        ],
    );

    let tolerance = Tolerance::<f32>::default();
    let check = |actual: &[f32], expected: TensorData| {
        TensorData::new(actual.to_vec(), expected.shape().clone())
            .assert_approx_eq::<f32>(&expected, tolerance);
    };
    check(&actual[0], TensorData::from([all]));
    check(&actual[1], TensorData::from([all_empty_axes]));
    check(&actual[2], axis1.to_data());
    check(&actual[3], last_axis.to_data());
    check(&actual[4], axis0.to_data());
    check(&actual[5], axes02.to_data());
}
