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
class PointerType;
class MemDescType;
namespace detail {
struct PointerTypeStorage;
} // namespace detail
class PointerType : public ::mlir::Type::TypeBase<PointerType, ::mlir::Type, detail::PointerTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tt.ptr";
  static PointerType get(Type pointeeType, int addressSpace);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ptr"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getPointeeType() const;
  int getAddressSpace() const;
};
namespace detail {
struct MemDescTypeStorage;
} // namespace detail
class MemDescType : public ::mlir::Type::TypeBase<MemDescType, ::mlir::Type, detail::MemDescTypeStorage, ::mlir::ShapedType::Trait> {
public:
  using Base::Base;
  MemDescType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                        Type elementType) const {
      return MemDescType::get(shape.value_or(getShape()), elementType, getEncoding());
  }

  bool hasRank() const { return true; }
  static constexpr ::llvm::StringLiteral name = "tt.memdesc";
  static MemDescType get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, Type elementType, Attribute encoding, bool mutable_memory);
  static MemDescType get(llvm::ArrayRef<int64_t> shape, Type elementType, Attribute encoding);
  static MemDescType get(llvm::ArrayRef<int64_t> shape, Type elementType, Attribute encoding, bool mutableMemory);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"memdesc"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  Type getElementType() const;
  Attribute getEncoding() const;
  bool getMutableMemory() const;
};
} // namespace triton
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::PointerType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::MemDescType)

#endif  // GET_TYPEDEF_CLASSES

