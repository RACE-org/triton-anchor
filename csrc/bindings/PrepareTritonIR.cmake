file(READ "${INPUT_FILE}" TRITON_IR_SOURCE)

string(REPLACE
  "#include \"third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h\"\n"
  ""
  TRITON_IR_SOURCE
  "${TRITON_IR_SOURCE}"
)

string(REPLACE
  "                    ::mlir::triton::proton::ProtonDialect, LLVM::LLVMDialect,\n"
  "                    LLVM::LLVMDialect,\n"
  TRITON_IR_SOURCE
  "${TRITON_IR_SOURCE}"
)

string(REPLACE [=[           })
      // Proton Ops
      .def("create_proton_record",
           [](TritonOpBuilder &self, bool isStart, int32_t regionId) -> void {
             self.create<mlir::triton::proton::RecordOp>(isStart, regionId);
           });]=]
  [=[           });]=]
  TRITON_IR_SOURCE
  "${TRITON_IR_SOURCE}"
)

get_filename_component(OUTPUT_DIR "${OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
file(WRITE "${OUTPUT_FILE}" "${TRITON_IR_SOURCE}")
