/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::triton::PointerType,
::mlir::triton::MemDescType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::triton::PointerType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::PointerType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::MemDescType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::MemDescType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::mlir::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::mlir::LogicalResult>(def)    .Case<::mlir::triton::PointerType>([&](auto t) {
      printer << ::mlir::triton::PointerType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::MemDescType>([&](auto t) {
      printer << ::mlir::triton::MemDescType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace triton {
namespace detail {
struct PointerTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type, int>;
  PointerTypeStorage(Type pointeeType, int addressSpace) : pointeeType(std::move(pointeeType)), addressSpace(std::move(addressSpace)) {}

  KeyTy getAsKey() const {
    return KeyTy(pointeeType, addressSpace);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (pointeeType == std::get<0>(tblgenKey)) && (addressSpace == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static PointerTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto pointeeType = std::move(std::get<0>(tblgenKey));
    auto addressSpace = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<PointerTypeStorage>()) PointerTypeStorage(std::move(pointeeType), std::move(addressSpace));
  }

  Type pointeeType;
  int addressSpace;
};
} // namespace detail
PointerType PointerType::get(Type pointeeType, int addressSpace) {
  return Base::get(pointeeType.getContext(), pointeeType, addressSpace);
}

Type PointerType::getPointeeType() const {
  return getImpl()->pointeeType;
}

int PointerType::getAddressSpace() const {
  return getImpl()->addressSpace;
}

} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::PointerType)
namespace mlir {
namespace triton {
namespace detail {
struct MemDescTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, Type, Attribute, bool>;
  MemDescTypeStorage(::llvm::ArrayRef<int64_t> shape, Type elementType, Attribute encoding, bool mutable_memory) : shape(std::move(shape)), elementType(std::move(elementType)), encoding(std::move(encoding)), mutable_memory(std::move(mutable_memory)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType, encoding, mutable_memory);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey)) && (encoding == std::get<2>(tblgenKey)) && (mutable_memory == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static MemDescTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    auto encoding = std::move(std::get<2>(tblgenKey));
    auto mutable_memory = std::move(std::get<3>(tblgenKey));
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<MemDescTypeStorage>()) MemDescTypeStorage(std::move(shape), std::move(elementType), std::move(encoding), std::move(mutable_memory));
  }

  ::llvm::ArrayRef<int64_t> shape;
  Type elementType;
  Attribute encoding;
  bool mutable_memory;
};
} // namespace detail
MemDescType MemDescType::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, Type elementType, Attribute encoding, bool mutable_memory) {
  return Base::get(context, std::move(shape), std::move(elementType), std::move(encoding), std::move(mutable_memory));
}

MemDescType MemDescType::get(llvm::ArrayRef<int64_t> shape, Type elementType, Attribute encoding) {
  return Base::get(elementType.getContext(), shape, elementType, encoding, /*mutableMemory=*/false);
}

MemDescType MemDescType::get(llvm::ArrayRef<int64_t> shape, Type elementType, Attribute encoding, bool mutableMemory) {
  return Base::get(elementType.getContext(), shape, elementType, encoding, mutableMemory);
}

::llvm::ArrayRef<int64_t> MemDescType::getShape() const {
  return getImpl()->shape;
}

Type MemDescType::getElementType() const {
  return getImpl()->elementType;
}

Attribute MemDescType::getEncoding() const {
  return getImpl()->encoding;
}

bool MemDescType::getMutableMemory() const {
  return getImpl()->mutable_memory;
}

} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::MemDescType)
namespace mlir {
namespace triton {

/// Parse a type registered to this dialect.
::mlir::Type TritonDialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void TritonDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}
} // namespace triton
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES

