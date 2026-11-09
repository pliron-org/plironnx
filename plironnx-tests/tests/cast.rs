// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/cast/mod.rs (burn-onnx repo).

#[allow(dead_code, clippy::type_complexity)]
mod cast_model {
    include!(concat!(env!("OUT_DIR"), "/model/cast.rs"));
}

use burn::tensor::{Bool, Device, Int, Tensor, TensorData};

mod common;
#[path = "common/jit.rs"]
mod jit;
mod utils;

#[test]
fn cast_ir() {
    utils::assert_ir_snapshots(
        "cast/cast.onnx",
        [
            "cast_onnx.expect.plir",
            "cast_tensor.expect.plir",
            "cast_llvm.expect.plir",
        ],
    );
}

#[test]
fn cast() {
    let device: Device = Default::default();
    let input_bool = Tensor::<2, Bool>::from_bool(TensorData::from([[true], [true]]), &device);
    let input_int = Tensor::<2, Int>::from_ints([[1], [1]], &device);
    let input_float = Tensor::<2>::from_floats([[1f32], [1.]], &device);
    let input_scalar = 1f32;

    let expected = cast_model::Model::new(&device).forward(
        input_bool.clone(),
        input_int.clone(),
        input_float.clone(),
        input_scalar,
    );

    let input_data = [
        input_bool.to_data(),
        input_int.to_data(),
        input_float.to_data(),
    ];
    let input_descriptors = [
        jit::build_input_descriptor::<bool>(&input_data[0]),
        jit::build_input_descriptor::<i32>(&input_data[1]),
        jit::build_input_descriptor::<f32>(&input_data[2]),
    ];
    let inputs = [
        input_descriptors[0].as_ptr(),
        input_descriptors[1].as_ptr(),
        input_descriptors[2].as_ptr(),
        (&raw const input_scalar).cast::<u8>(),
    ];

    let mut tensor_outputs: [Vec<u8>; 9] =
        std::array::from_fn(|_| jit::build_empty_output_descriptor(2));
    let mut scalar_output = 0i32;
    let outputs: Vec<*mut u8> = tensor_outputs
        .iter_mut()
        .map(|descriptor| descriptor.as_mut_ptr())
        .chain(std::iter::once((&raw mut scalar_output).cast::<u8>()))
        .collect();
    jit::run_packed("cast/cast.onnx", &inputs, &outputs);

    assert_eq!(
        jit::read_output::<bool>(&tensor_outputs[0], &[2, 1]),
        expected.0.to_data().as_slice::<bool>().unwrap()
    );
    assert_eq!(
        jit::read_output::<i32>(&tensor_outputs[1], &[2, 1]),
        expected.1.to_data().as_slice::<i32>().unwrap()
    );
    assert_eq!(
        jit::read_output::<f32>(&tensor_outputs[2], &[2, 1]),
        expected.2.to_data().as_slice::<f32>().unwrap()
    );
    assert_eq!(
        jit::read_output::<bool>(&tensor_outputs[3], &[2, 1]),
        expected.3.to_data().as_slice::<bool>().unwrap()
    );
    assert_eq!(
        jit::read_output::<i32>(&tensor_outputs[4], &[2, 1]),
        expected.4.to_data().as_slice::<i32>().unwrap()
    );
    assert_eq!(
        jit::read_output::<f32>(&tensor_outputs[5], &[2, 1]),
        expected.5.to_data().as_slice::<f32>().unwrap()
    );
    assert_eq!(
        jit::read_output::<bool>(&tensor_outputs[6], &[2, 1]),
        expected.6.to_data().as_slice::<bool>().unwrap()
    );
    assert_eq!(
        jit::read_output::<i32>(&tensor_outputs[7], &[2, 1]),
        expected.7.to_data().as_slice::<i32>().unwrap()
    );
    assert_eq!(
        jit::read_output::<f32>(&tensor_outputs[8], &[2, 1]),
        expected.8.to_data().as_slice::<f32>().unwrap()
    );
    assert_eq!(scalar_output, expected.9);
}
