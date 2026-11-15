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

/// Check that Relu of tensors and of scalars gives `max(x, 0)` for each element:
/// - NaN must give NaN
/// - `-0.0` and `-inf` must give `+0.0`
/// - signed integers must compare as signed values
#[test]
fn relu_special_values() {
    let ir = r#"
builtin.module @onnx_module
{
  ^block1():
    builtin.func @model: builtin.function <(tensor.ranked <8 : builtin.fp32 >, tensor.ranked <4 : builtin.integer i32>, builtin.fp32, builtin.integer i32) -> (tensor.ranked <8 : builtin.fp32 >, tensor.ranked <4 : builtin.integer i32>, builtin.fp32, builtin.integer i32)>
    {
      ^entry(f: tensor.ranked <8 : builtin.fp32 >, i: tensor.ranked <4 : builtin.integer i32>, s: builtin.fp32, t: builtin.integer i32):
        rf = onnx.relu f : tensor.ranked <8 : builtin.fp32 >;
        ri = onnx.relu i : tensor.ranked <4 : builtin.integer i32>;
        rs = onnx.relu s : builtin.fp32;
        rt = onnx.relu t : builtin.integer i32;
        onnx.return rf, ri, rs, rt
    }
}
"#;
    let f = [
        f32::NAN,
        -0.0,
        0.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
        -1.5,
        2.5,
        -f32::NAN,
    ];
    let i = [-5, 0, 7, i32::MIN];
    let (s, t) = (-3.5f32, 9i32);

    let f_data = TensorData::new(f.to_vec(), [8]);
    let i_data = TensorData::new(i.to_vec(), [4]);
    let f_descriptor = jit::build_input_descriptor::<f32>(&f_data);
    let i_descriptor = jit::build_input_descriptor::<i32>(&i_data);
    let mut rf = jit::build_empty_output_descriptor(1);
    let mut ri = jit::build_empty_output_descriptor(1);
    let (mut rs, mut rt) = (f32::NAN, 0i32);
    jit::run_packed_ir(
        ir,
        &[
            f_descriptor.as_ptr(),
            i_descriptor.as_ptr(),
            (&raw const s).cast(),
            (&raw const t).cast(),
        ],
        &[
            rf.as_mut_ptr(),
            ri.as_mut_ptr(),
            (&raw mut rs).cast(),
            (&raw mut rt).cast(),
        ],
    );

    let actual = jit::read_output::<f32>(&rf, &[8]);
    let expected = [f32::NAN, 0.0, 0.0, 0.0, f32::INFINITY, 0.0, 2.5, f32::NAN];
    for (idx, (a, e)) in actual.iter().zip(&expected).enumerate() {
        // NaN must match NaN, and every other element must match bit for bit.
        assert!(
            (a.is_nan() && e.is_nan()) || a.to_bits() == e.to_bits(),
            "element {idx}: got {a:?}, expected {e:?}"
        );
    }
    assert_eq!(jit::read_output::<i32>(&ri, &[4]), [0, 0, 7, 0]);
    assert_eq!(rs.to_bits(), 0.0f32.to_bits());
    assert_eq!(rt, 9);
}
