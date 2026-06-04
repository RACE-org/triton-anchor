/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace triton {
namespace gpu {
class TokenType;
class AsyncTokenType;
namespace detail {
struct TokenTypeStorage;
} // namespace detail
class TokenType : public ::mlir::Type::TypeBase<TokenType, ::mlir::Type, detail::TokenTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "triton_gpu.token";
  static TokenType get(::mlir::MLIRContext *context, unsigned type);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"token"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int32_t getType() const;
};
class AsyncTokenType : public ::mlir::Type::TypeBase<AsyncTokenType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "triton_gpu.async.token";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"async.token"};
  }

};
} // namespace gpu
} // namespace triton
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::TokenType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::AsyncTokenType)

#endif  // GET_TYPEDEF_CLASSES

