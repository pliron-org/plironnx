// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

use std::path::PathBuf;

use cargo_metadata::MetadataCommand;

/// Locate the `tests/` directory of the `onnx-tests` crate (from the burn-onnx
/// project) that we depend on. This is where the `.onnx` test models used
/// by burn's own ONNX import tests live, one subdirectory per operator (e.g.
/// `tests/add/add.onnx`).
pub fn onnx_tests_dir() -> PathBuf {
    let metadata = MetadataCommand::new()
        .exec()
        .expect("failed to run `cargo metadata`");

    let pkg = metadata
        .packages
        .iter()
        .find(|pkg| pkg.name.as_str() == "onnx-tests")
        .expect("onnx-tests package not found; is it in [dev-dependencies]?");

    pkg.manifest_path
        .parent()
        .expect("onnx-tests manifest_path has no parent directory")
        .join("tests")
        .into_std_path_buf()
}
