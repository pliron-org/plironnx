// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! `onnx` dialect IR tests.

use burn::tensor::{Device, Int, Tensor, TensorData};

mod common;
#[path = "common/jit.rs"]
mod jit;

/// Check that the product of two vectors gives a rank-0 tensor with their dot product.
#[test]
fn matmul_vector_vector() {
    let ir = r#"
builtin.module @onnx_module
{
  ^block1():
    builtin.func @model: builtin.function <(tensor.ranked <4 : builtin.fp32 >, tensor.ranked <4 : builtin.fp32 >) -> (tensor.ranked < : builtin.fp32 >)>
    {
      ^entry(a: tensor.ranked <4 : builtin.fp32 >, b: tensor.ranked <4 : builtin.fp32 >):
        dot = onnx.matmul a, b : tensor.ranked < : builtin.fp32 >;
        onnx.return dot
    }
}
"#;
    let device: Device = Default::default();
    let a = Tensor::<1, Int>::arange(0..4, &device).float();
    let b = Tensor::<1, Int>::arange(1..5, &device).float();
    let expected: f32 = (a.clone() * b.clone()).sum().into_scalar();

    let input_data = [a.to_data(), b.to_data()];
    let input_descriptors = input_data
        .each_ref()
        .map(jit::build_input_descriptor::<f32>);
    let mut output = jit::build_empty_output_descriptor(0);
    jit::run_packed_ir(
        ir,
        &input_descriptors
            .each_ref()
            .map(|descriptor| descriptor.as_ptr()),
        &[output.as_mut_ptr()],
    );

    assert_eq!(jit::read_output::<f32>(&output, &[]), vec![expected]);
}

/// Check that integer division of tensors and of scalars truncates toward zero.
#[test]
fn div_integer() {
    let ir = r#"
builtin.module @onnx_module
{
  ^block1():
    builtin.func @model: builtin.function <(tensor.ranked <6 : builtin.integer i32>, tensor.ranked <6 : builtin.integer i32>, builtin.integer i32, builtin.integer i32) -> (tensor.ranked <6 : builtin.integer i32>, builtin.integer i32)>
    {
      ^entry(a: tensor.ranked <6 : builtin.integer i32>, b: tensor.ranked <6 : builtin.integer i32>, s: builtin.integer i32, t: builtin.integer i32):
        q = onnx.div a, b : tensor.ranked <6 : builtin.integer i32>;
        r = onnx.div s, t : builtin.integer i32;
        onnx.return q, r
    }
}
"#;
    let a = [7, -7, 7, -7, 0, 9];
    let b = [2, 2, -2, -2, 3, 3];
    let (s, t) = (-9i32, 4i32);

    let a_data = TensorData::new(a.to_vec(), [6]);
    let b_data = TensorData::new(b.to_vec(), [6]);
    let a_descriptor = jit::build_input_descriptor::<i32>(&a_data);
    let b_descriptor = jit::build_input_descriptor::<i32>(&b_data);
    let mut q = jit::build_empty_output_descriptor(1);
    let mut r = 0i32;
    jit::run_packed_ir(
        ir,
        &[
            a_descriptor.as_ptr(),
            b_descriptor.as_ptr(),
            (&raw const s).cast(),
            (&raw const t).cast(),
        ],
        &[q.as_mut_ptr(), (&raw mut r).cast()],
    );

    // Rust integer division also truncates toward zero.
    let expected: Vec<i32> = a.iter().zip(&b).map(|(a, b)| a / b).collect();
    assert_eq!(jit::read_output::<i32>(&q, &[6]), expected);
    assert_eq!(r, s / t);
}
