// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Test from onnx-tests' tests/cast/mod.rs (burn-onnx repo).

#[allow(dead_code, clippy::type_complexity)]
mod cast_model {
    include!(concat!(env!("OUT_DIR"), "/model/cast.rs"));
}

use burn::tensor::{Bool, Device, Int, Tensor, TensorData};
use pliron::{context::Context, op::verify_op, result::ExpectOk};
use pliron_llvm::llvm_sys::lljit::SimpleJIT;
use pliron_tensor::tensor::runtime_utils::TensorDesciptor;
use plironnx::{
    burn_onnx_ir::parse_onnx_file,
    onnx::{entrypoint_abi::compile_packed_entrypoint, from_onnx_graph::build_module},
};

mod common;
mod utils;

fn read_tensor_output<T: Copy>(descriptor: &[u8]) -> Vec<T> {
    unsafe {
        let descriptor =
            TensorDesciptor::from_ir_descriptor(descriptor.as_ptr(), 2, size_of::<T>());
        assert_eq!(descriptor.sizes(), &[2, 1]);
        let mut values = Vec::new();
        descriptor.copy_to_vec(&mut values);
        values
    }
}

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

    let bool_data = input_bool.to_data();
    let int_data = input_int.to_data();
    let float_data = input_float.to_data();
    let bool_values = bool_data.as_slice::<bool>().expect("bool input");
    let int_values = int_data.as_slice::<i32>().expect("i32 input");
    let float_values = float_data.as_slice::<f32>().expect("f32 input");
    let input_descriptors = [
        TensorDesciptor::new(vec![2, 1], size_of::<bool>(), bool_values.as_ptr().cast())
            .build_ir_descriptor(),
        TensorDesciptor::new(vec![2, 1], size_of::<i32>(), int_values.as_ptr().cast())
            .build_ir_descriptor(),
        TensorDesciptor::new(vec![2, 1], size_of::<f32>(), float_values.as_ptr().cast())
            .build_ir_descriptor(),
    ];
    let inputs = [
        input_descriptors[0].as_ptr(),
        input_descriptors[1].as_ptr(),
        input_descriptors[2].as_ptr(),
        (&raw const input_scalar).cast::<u8>(),
    ];

    let graph = parse_onnx_file(common::onnx_tests_dir().join("cast/cast.onnx"))
        .expect("failed to parse cast.onnx");
    let ctx = &mut Context::new();
    let module = build_module(ctx, &graph, "onnx_module", "model").expect_ok(ctx);
    verify_op(&module, ctx).expect_ok(ctx);
    let (llvm_ctx, llvm_module, abi) =
        compile_packed_entrypoint(ctx, module, "model", "main").expect_ok(ctx);
    assert_eq!((abi.num_inputs, abi.num_outputs), (4, 10));
    llvm_module.verify().expect("generated LLVM IR must verify");
    let jit = SimpleJIT::new(llvm_ctx, llvm_module).expect("failed to create JIT");

    let mut tensor_outputs = [[0u8; (3 + 2 * 2) * size_of::<usize>()]; 9];
    let mut scalar_output = 0i32;
    let outputs: Vec<*mut u8> = tensor_outputs
        .iter_mut()
        .map(|descriptor| descriptor.as_mut_ptr())
        .chain(std::iter::once((&raw mut scalar_output).cast::<u8>()))
        .collect();
    let function = unsafe {
        jit.lookup_symbol::<extern "C" fn(*const *const u8, *const *mut u8) -> ()>("main")
    }
    .expect("failed to find compiled main function");
    function(inputs.as_ptr(), outputs.as_ptr());

    assert_eq!(
        read_tensor_output::<bool>(&tensor_outputs[0]),
        expected.0.to_data().as_slice::<bool>().unwrap()
    );
    assert_eq!(
        read_tensor_output::<i32>(&tensor_outputs[1]),
        expected.1.to_data().as_slice::<i32>().unwrap()
    );
    assert_eq!(
        read_tensor_output::<f32>(&tensor_outputs[2]),
        expected.2.to_data().as_slice::<f32>().unwrap()
    );
    assert_eq!(
        read_tensor_output::<bool>(&tensor_outputs[3]),
        expected.3.to_data().as_slice::<bool>().unwrap()
    );
    assert_eq!(
        read_tensor_output::<i32>(&tensor_outputs[4]),
        expected.4.to_data().as_slice::<i32>().unwrap()
    );
    assert_eq!(
        read_tensor_output::<f32>(&tensor_outputs[5]),
        expected.5.to_data().as_slice::<f32>().unwrap()
    );
    assert_eq!(
        read_tensor_output::<bool>(&tensor_outputs[6]),
        expected.6.to_data().as_slice::<bool>().unwrap()
    );
    assert_eq!(
        read_tensor_output::<i32>(&tensor_outputs[7]),
        expected.7.to_data().as_slice::<i32>().unwrap()
    );
    assert_eq!(
        read_tensor_output::<f32>(&tensor_outputs[8]),
        expected.8.to_data().as_slice::<f32>().unwrap()
    );
    assert_eq!(scalar_output, expected.9);
}
