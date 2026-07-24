// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Tests from onnx-tests' tests/add/mod.rs (burn-onnx repo)

#[allow(dead_code)]
mod add_model {
    include!(concat!(env!("OUT_DIR"), "/model/add.rs"));
}

use burn::tensor::{Device, Tensor};

type Backend = burn::backend::Flex;

/// Reference execution: generated model from burn.
fn run_reference(input: Tensor<Backend, 4>, scalar: f64) -> Tensor<Backend, 4> {
    let model = add_model::Model::<Backend>::default();
    model.forward(input, scalar)
}

/// TODO: replace with plironnx's compiled-and-run output once that pipeline
/// exists. For now it just re-runs the reference model, so this test
/// compares burn's output with itself.
fn run_plironnx(input: Tensor<Backend, 4>, scalar: f64) -> Tensor<Backend, 4> {
    run_reference(input, scalar)
}

#[test]
fn add() {
    let device: Device<Backend> = Default::default();

    // Inputs from onnx-tests/tests/add/mod.rs : add_scalar_to_tensor_and_tensor_to_tensor
    let input = Tensor::<Backend, 4>::from_floats([[[[1., 2., 3., 4.]]]], &device);
    let scalar = 2f64;

    let expected = run_reference(input.clone(), scalar);
    let actual = run_plironnx(input, scalar);

    actual.to_data().assert_eq(&expected.to_data(), true);
}
