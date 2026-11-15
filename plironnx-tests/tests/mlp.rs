// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test of the MNIST-sized multilayer perceptron in models/mlp (see models/mlp/mlp.py).

#[allow(dead_code)]
mod mlp_model {
    include!(concat!(env!("OUT_DIR"), "/model/mlp.rs"));
}

use burn::tensor::{Device, Int, Tensor, Tolerance};

mod common;
#[path = "common/jit.rs"]
mod jit;

/// Check that the logits of a batch of three 28x28 images match Burn.
#[test]
fn mlp() {
    let device: Device = Default::default();
    let len = 3 * 28 * 28;
    let input = Tensor::<1, Int>::arange(0..len, &device)
        .float()
        .mul_scalar(0.37)
        .sin()
        .reshape([3, 1, 28, 28]);
    let expected = mlp_model::Model::default().forward(input.clone());
    let actual = jit::run_single_f32(
        common::local_models_dir().join("mlp/mlp.onnx"),
        &input,
        [3, 10],
    );
    actual
        .to_data()
        .assert_approx_eq(&expected.to_data(), Tolerance::<f32>::default());
}
