// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

//! Lower ONNX entry points to a stable, arity-independent host ABI.

use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{
            CallOpCallable, OneRegionInterface, OneResultInterface, SingleBlockRegionInterface,
            SymbolOpInterface,
        },
        ops::{FuncOp as BuiltinFuncOp, ModuleOp},
        type_interfaces::FunctionTypeInterface,
        types::{FunctionType, UnitType},
    },
    context::Context,
    identifier::Identifier,
    input_err_noloc,
    irbuild::{
        dialect_conversion::apply_dialect_conversion,
        inserter::{IRInserter, Inserter},
        listener::DummyListener,
        rewriter::{IRRewriter, Rewriter},
    },
    linked_list::ContainsLinkedList,
    op::{Op, verify_op},
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, TypedHandle},
    value::Value,
};
use pliron_common_dialects::cf::to_llvm::CFToLLVM;
use pliron_llvm::{
    builtin_to_llvm::BuiltinToLLVMConversion,
    llvm_sys::core::{LLVMContext, LLVMModule},
    ops::{CallOp, FuncOp, GepIndex, GetElementPtrOp, LoadOp, ReturnOp, StoreOp},
    types::{FuncType, PointerType, VoidType},
};
use pliron_tensor::{
    memref::conversions::MemrefToCF,
    tensor::{bufferize::bufferize, memory_management::MallocFreeTMM},
};

use crate::onnx::{conversions::OnnxToTensor, ops::ReturnOp as OnnxReturnOp};

type OpInserter = IRInserter<DummyListener>;
type OpRewriter = IRRewriter<DummyListener>;

#[derive(Debug, thiserror::Error)]
pub enum EntrypointAbiErr {
    #[error("The module has no function with the name `{0}`")]
    MissingFunction(String),
}

/// The number of input and output slots of a packed entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntrypointAbi {
    pub num_inputs: usize,
    pub num_outputs: usize,
}

/// Compile an ONNX module and export an arity-independent packed entry point.
///
/// The exported function has the native signature
/// `void entry(void **inputs, void **outputs)`. Array slots follow the graph's
/// input and output order and point to storage for the corresponding typed value.
///
/// [`EntrypointAbi`] conveys the number of slots in each array.
pub fn compile_packed_entrypoint(
    ctx: &mut Context,
    module: ModuleOp,
    model_name: &str,
    entry_name: &str,
) -> Result<(LLVMContext, LLVMModule, EntrypointAbi)> {
    let root = module.get_operation();
    apply_dialect_conversion(ctx, &mut OnnxToTensor, root)?;

    let model = find_func::<BuiltinFuncOp>(ctx, module, &Identifier::try_from(model_name)?)?;
    let abi = lower_model_outputs(ctx, model)?;

    apply_dialect_conversion(ctx, &mut BuiltinToLLVMConversion, root)?;
    bufferize(&mut MallocFreeTMM, root, ctx)?;
    apply_dialect_conversion(ctx, &mut MemrefToCF, root)?;
    apply_dialect_conversion(ctx, &mut CFToLLVM, root)?;

    add_packed_wrapper(ctx, module, model_name, entry_name, abi)?;
    verify_op(&module, ctx)?;

    let llvm_ctx = LLVMContext::default();
    let llvm_module = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module)?;
    Ok((llvm_ctx, llvm_module, abi))
}

/// Find the function that defines `name` in the single body block of `module`.
fn find_func<T: Op + SymbolOpInterface>(
    ctx: &Context,
    module: ModuleOp,
    name: &Identifier,
) -> Result<T> {
    let func = module
        .get_body(ctx, 0)
        .deref(ctx)
        .iter(ctx)
        .filter_map(|op| Operation::get_op::<T>(op, ctx))
        .find(|func| &func.get_symbol_name(ctx) == name);
    match func {
        Some(func) => Ok(func),
        None => input_err_noloc!(EntrypointAbiErr::MissingFunction(name.to_string())),
    }
}

/// Change model entry point to return through output-storage pointers.
///
/// This runs after ONNX operation lowering and before builtin-to-LLVM conversion
/// and tensor bufferization.
///
/// ```text
/// // Before
/// builtin.func @model(%image: tensor, %scale: f32) -> (tensor, f32) {
///   ...
///   onnx.return %result, %score
/// }
///
/// // After
/// builtin.func @model(
///     %image: tensor, %scale: f32, %result_out: ptr, %score_out: ptr
/// ) -> unit {
///   ...
///   llvm.store %result, %result_out
///   llvm.store %score, %score_out
///   llvm.return
/// }
/// ```
pub fn lower_model_outputs(ctx: &mut Context, func: BuiltinFuncOp) -> Result<EntrypointAbi> {
    let func_ty = TypedHandle::<FunctionType>::from_handle(func.get_type(ctx), ctx)?;
    let (input_types, num_outputs) = {
        let func_ty = func_ty.deref(ctx);
        (func_ty.arg_types(), func_ty.res_types().len())
    };
    let abi = EntrypointAbi {
        num_inputs: input_types.len(),
        num_outputs,
    };

    let ptr = TypeHandle::from(PointerType::get(ctx, 0));
    let mut argument_types = input_types;
    argument_types.extend(core::iter::repeat_n(ptr, abi.num_outputs));
    let new_ty = FunctionType::get(ctx, argument_types, vec![UnitType::get(ctx).into()]);
    func.set_attr_builtin_func_type(ctx, TypeAttr::new(new_ty.into()));

    // Add one output-storage pointer argument for each result.
    let entry = func.get_entry_block(ctx);
    assert_eq!(
        entry.deref(ctx).get_num_arguments(),
        abi.num_inputs,
        "the entry block arguments must agree with the function inputs"
    );
    let output_ptrs = (0..abi.num_outputs)
        .map(|_| {
            let arg_idx = BasicBlock::push_argument(entry, ctx, ptr);
            entry.deref(ctx).get_argument(arg_idx)
        })
        .collect::<Vec<_>>();

    // The output pointers are entry block arguments, thus they dominate all the
    // blocks of the function. Replace the returns in each of those blocks.
    let returns = func
        .get_region(ctx)
        .deref(ctx)
        .iter(ctx)
        .flat_map(|block| {
            block
                .deref(ctx)
                .iter(ctx)
                .filter_map(|op| Operation::get_op::<OnnxReturnOp>(op, ctx))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut rewriter = OpRewriter::default();
    for return_op in returns {
        let op = return_op.get_operation();
        let outputs = op.deref(ctx).operands().collect::<Vec<_>>();
        assert_eq!(
            outputs.len(),
            abi.num_outputs,
            "the return operands must agree with the function results"
        );
        rewriter.set_insertion_point_before_operation(op);
        for (output, output_ptr) in outputs.into_iter().zip(output_ptrs.iter().copied()) {
            let store = StoreOp::new(ctx, output, output_ptr);
            rewriter.append_op(ctx, &store);
        }
        let ret = ReturnOp::new(ctx, None);
        rewriter.append_op(ctx, &ret);
        rewriter.erase_operation(ctx, op);
    }
    Ok(abi)
}

/// Read the pointer that the packed slot `base[index]` holds.
fn load_slot_pointer(
    ctx: &mut Context,
    inserter: &mut OpInserter,
    base: Value,
    index: usize,
) -> Value {
    let ptr = TypeHandle::from(PointerType::get(ctx, 0));
    let index = u32::try_from(index).expect("slot index exceeds u32");
    let slot = GetElementPtrOp::new(ctx, base, vec![GepIndex::Constant(index)], ptr);
    inserter.append_op(ctx, &slot);
    let value_ptr = LoadOp::new(ctx, slot.get_result(ctx), ptr);
    inserter.append_op(ctx, &value_ptr);
    value_ptr.get_result(ctx)
}

/// Add an exported `void entry(void **inputs, void **outputs)` wrapper.
///
/// Each input slot points to storage containing the statically typed input.
/// Each output slot points to writable storage for its result descriptor or scalar.
///
/// ```text
/// // Existing typed function
/// llvm.func @model(%image: memref, %scale: f32, %result_out: ptr) -> void
///
/// // Generated packed wrapper
/// llvm.func @entry(%inputs: ptr, %outputs: ptr) -> void {
///   %image_ptr = load (gep %inputs[0])
///   %image = load %image_ptr : memref
///   %scale_ptr = load (gep %inputs[1])
///   %scale = load %scale_ptr : f32
///   %result_out = load (gep %outputs[0])
///   llvm.call @model(%image, %scale, %result_out)
///   llvm.return
/// }
/// ```
pub fn add_packed_wrapper(
    ctx: &mut Context,
    module: ModuleOp,
    model_name: &str,
    entry_name: &str,
    abi: EntrypointAbi,
) -> Result<()> {
    let model_name = Identifier::try_from(model_name)?;
    let model = find_func::<FuncOp>(ctx, module, &model_name)?;
    let model_ty = model.get_type(ctx);
    let model_arg_types = model_ty.deref(ctx).arg_types();
    assert_eq!(model_arg_types.len(), abi.num_inputs + abi.num_outputs);

    let ptr = TypeHandle::from(PointerType::get(ctx, 0));
    let wrapper_ty = FuncType::get(ctx, VoidType::get(ctx).to_handle(), vec![ptr, ptr], false);
    let wrapper = FuncOp::new(ctx, Identifier::try_from(entry_name)?, wrapper_ty);
    let entry = wrapper.get_or_create_entry_block(ctx);
    let (packed_inputs, packed_outputs) = {
        let entry = entry.deref(ctx);
        (entry.get_argument(0), entry.get_argument(1))
    };

    let mut inserter = OpInserter::new_at_block_end(entry);
    let mut call_args = Vec::with_capacity(model_arg_types.len());
    // An input slot points to storage that holds the value. Load the value.
    for (index, input_type) in model_arg_types.iter().take(abi.num_inputs).enumerate() {
        let value_ptr = load_slot_pointer(ctx, &mut inserter, packed_inputs, index);
        let value = LoadOp::new(ctx, value_ptr, *input_type);
        inserter.append_op(ctx, &value);
        call_args.push(value.get_result(ctx));
    }
    // An output slot points to storage for the result. Pass that pointer on.
    for index in 0..abi.num_outputs {
        call_args.push(load_slot_pointer(ctx, &mut inserter, packed_outputs, index));
    }

    let call = CallOp::new(ctx, CallOpCallable::Direct(model_name), model_ty, call_args);
    inserter.append_op(ctx, &call);
    let ret = ReturnOp::new(ctx, None);
    inserter.append_op(ctx, &ret);
    module.append_operation(ctx, wrapper.get_operation(), 0);
    Ok(())
}
