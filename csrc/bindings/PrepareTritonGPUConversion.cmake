file(READ "${INPUT_FILE}" TRITON_CONVERSION_SOURCE)

string(REPLACE
  "#include \"third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h\"\n"
  ""
  TRITON_CONVERSION_SOURCE
  "${TRITON_CONVERSION_SOURCE}"
)

string(REPLACE [=[// Proton patterns
// NOTE: Because Proton's inputs are scalars and not tensors this conversion
// isn't strictly necessary however you could envision a case where we pass in
// tensors in for Triton object specific tracing operations in which case we
// would need to fill in the OpConversionPattern
void populateProtonPatterns(TritonGPUTypeConverter &typeConverter,
                            RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GenericOpPattern<triton::proton::RecordOp>>(typeConverter,
                                                           context);
}
]=]
  ""
  TRITON_CONVERSION_SOURCE
  "${TRITON_CONVERSION_SOURCE}"
)

string(REPLACE
  "    populateProtonPatterns(typeConverter, patterns);\n"
  ""
  TRITON_CONVERSION_SOURCE
  "${TRITON_CONVERSION_SOURCE}"
)

get_filename_component(OUTPUT_DIR "${OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
file(WRITE "${OUTPUT_FILE}" "${TRITON_CONVERSION_SOURCE}")
