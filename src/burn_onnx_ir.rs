// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The plironnx contributors

use std::path::Path;

use onnx_ir::{OnnxGraph, OnnxGraphBuilder};

use pliron::{input_error_noloc, result::Result};

/// Parse an ONNX model file into its [`OnnxGraph`] intermediate representation.
pub fn parse_onnx_file(path: impl AsRef<Path>) -> Result<OnnxGraph> {
    OnnxGraphBuilder::new()
        .parse_file(path)
        .map_err(|err| input_error_noloc!(err))
}
