/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonNvidiaGPUOps.td                                                *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp,
::mlir::triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp,
::mlir::triton::nvidia_gpu::BarrierExpectOp,
::mlir::triton::nvidia_gpu::ClusterArriveOp,
::mlir::triton::nvidia_gpu::ClusterWaitOp,
::mlir::triton::nvidia_gpu::DotAsyncOp,
::mlir::triton::nvidia_gpu::DotWaitOp,
::mlir::triton::nvidia_gpu::FenceAsyncSharedOp,
::mlir::triton::nvidia_gpu::InitBarrierOp,
::mlir::triton::nvidia_gpu::InvalBarrierOp,
::mlir::triton::nvidia_gpu::TMAStoreWait,
::mlir::triton::nvidia_gpu::WaitBarrierOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace nvidia_gpu {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::triton::PointerType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be Pointer type (`::mlir::triton::PointerType`) in Triton IR type system, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::triton::MemDescType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memory descriptor type (`::mlir::triton::MemDescType`) in Triton IR type system, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::TensorOrMemDesc>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be TensorOrMemDesc instance, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isFloat8E4M3FNUZ())) || ((elementType.isFloat8E5M2())) || ((elementType.isFloat8E5M2FNUZ())) || ((elementType.isF16())) || ((elementType.isBF16())) || ((elementType.isF32())) || ((elementType.isF64()))) || (((elementType.isSignlessInteger(1))) || ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64)))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of floating-point or integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::TensorOrMemDesc>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of TensorOrMemDesc instance, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TritonNvidiaGPUOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps0(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::triton::CacheModifierAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 1, 2, 3, 4, 5, 6";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps0(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::triton::EvictionPolicyAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 1, 2, 3";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::BoolAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: bool attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps4(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(1)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 1-bit signless integer attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps4(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps5(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::triton::InputPrecisionAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1, 2";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonNvidiaGPUOps5(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::AsyncTMACopyGlobalToLocalOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("triton_nvidia_gpu.async_tma_copy_global_to_local", odsAttrs.getContext());
}

AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::AsyncTMACopyGlobalToLocalOpGenericAdaptorBase(AsyncTMACopyGlobalToLocalOp op) : AsyncTMACopyGlobalToLocalOpGenericAdaptorBase(op->getDiscardableAttrDictionary(), op.getProperties(), op->getRegions()) {}

std::pair<unsigned, unsigned> AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true, false, false, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 4) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::triton::CacheModifierAttr AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::getCacheAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::triton::CacheModifierAttr>(getProperties().cache);
  return attr;
}

::mlir::triton::CacheModifier AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::getCache() {
  auto attr = getCacheAttr();
  return attr.getValue();
}

::mlir::triton::EvictionPolicyAttr AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::getEvictAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::triton::EvictionPolicyAttr>(getProperties().evict);
  return attr;
}

::mlir::triton::EvictionPolicy AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::getEvict() {
  auto attr = getEvictAttr();
  return attr.getValue();
}

::mlir::BoolAttr AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::getIsVolatileAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().isVolatile);
  return attr;
}

bool AsyncTMACopyGlobalToLocalOpGenericAdaptorBase::getIsVolatile() {
  auto attr = getIsVolatileAttr();
  return attr.getValue();
}

} // namespace detail
AsyncTMACopyGlobalToLocalOpAdaptor::AsyncTMACopyGlobalToLocalOpAdaptor(AsyncTMACopyGlobalToLocalOp op) : AsyncTMACopyGlobalToLocalOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult AsyncTMACopyGlobalToLocalOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_cache = getProperties().cache; (void)tblgen_cache;
  auto tblgen_evict = getProperties().evict; (void)tblgen_evict;
  auto tblgen_isVolatile = getProperties().isVolatile; (void)tblgen_isVolatile;

  if (tblgen_cache && !((::llvm::isa<::mlir::triton::CacheModifierAttr>(tblgen_cache))))
    return emitError(loc, "'triton_nvidia_gpu.async_tma_copy_global_to_local' op ""attribute 'cache' failed to satisfy constraint: allowed 32-bit signless integer cases: 1, 2, 3, 4, 5, 6");

  if (tblgen_evict && !((::llvm::isa<::mlir::triton::EvictionPolicyAttr>(tblgen_evict))))
    return emitError(loc, "'triton_nvidia_gpu.async_tma_copy_global_to_local' op ""attribute 'evict' failed to satisfy constraint: allowed 32-bit signless integer cases: 1, 2, 3");

  if (tblgen_isVolatile && !((::llvm::isa<::mlir::BoolAttr>(tblgen_isVolatile))))
    return emitError(loc, "'triton_nvidia_gpu.async_tma_copy_global_to_local' op ""attribute 'isVolatile' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AsyncTMACopyGlobalToLocalOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false, false, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 4) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange AsyncTMACopyGlobalToLocalOp::getCoordMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::LogicalResult AsyncTMACopyGlobalToLocalOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.cache;
       auto attr = dict.get("cache");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `cache` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.evict;
       auto attr = dict.get("evict");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `evict` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.isVolatile;
       auto attr = dict.get("isVolatile");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `isVolatile` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute AsyncTMACopyGlobalToLocalOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.cache;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("cache",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.evict;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("evict",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.isVolatile;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("isVolatile",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AsyncTMACopyGlobalToLocalOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.cache.getAsOpaquePointer()), 
    llvm::hash_value(prop.evict.getAsOpaquePointer()), 
    llvm::hash_value(prop.isVolatile.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> AsyncTMACopyGlobalToLocalOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "cache")
      return prop.cache;

    if (name == "evict")
      return prop.evict;

    if (name == "isVolatile")
      return prop.isVolatile;
  return std::nullopt;
}

void AsyncTMACopyGlobalToLocalOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "cache") {
       prop.cache = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.cache)>>(value);
       return;
    }

    if (name == "evict") {
       prop.evict = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.evict)>>(value);
       return;
    }

    if (name == "isVolatile") {
       prop.isVolatile = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.isVolatile)>>(value);
       return;
    }
}

void AsyncTMACopyGlobalToLocalOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.cache) attrs.append("cache", prop.cache);

    if (prop.evict) attrs.append("evict", prop.evict);

    if (prop.isVolatile) attrs.append("isVolatile", prop.isVolatile);
}

::mlir::LogicalResult AsyncTMACopyGlobalToLocalOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getCacheAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps0(attr, "cache", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getEvictAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(attr, "evict", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getIsVolatileAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps2(attr, "isVolatile", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult AsyncTMACopyGlobalToLocalOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.cache)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.evict)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.isVolatile)))
    return ::mlir::failure();
  return ::mlir::success();
}

void AsyncTMACopyGlobalToLocalOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.cache);

  writer.writeOptionalAttribute(prop.evict);

  writer.writeOptionalAttribute(prop.isVolatile);
}

::mlir::triton::CacheModifier AsyncTMACopyGlobalToLocalOp::getCache() {
  auto attr = getCacheAttr();
  return attr.getValue();
}

::mlir::triton::EvictionPolicy AsyncTMACopyGlobalToLocalOp::getEvict() {
  auto attr = getEvictAttr();
  return attr.getValue();
}

bool AsyncTMACopyGlobalToLocalOp::getIsVolatile() {
  auto attr = getIsVolatileAttr();
  return attr.getValue();
}

void AsyncTMACopyGlobalToLocalOp::setCache(::mlir::triton::CacheModifier attrValue) {
  getProperties().cache = ::mlir::triton::CacheModifierAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void AsyncTMACopyGlobalToLocalOp::setEvict(::mlir::triton::EvictionPolicy attrValue) {
  getProperties().evict = ::mlir::triton::EvictionPolicyAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void AsyncTMACopyGlobalToLocalOp::setIsVolatile(bool attrValue) {
  getProperties().isVolatile = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void AsyncTMACopyGlobalToLocalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value barrier, ::mlir::Value result, ::mlir::Value pred, ::mlir::triton::CacheModifierAttr cache, ::mlir::triton::EvictionPolicyAttr evict, ::mlir::BoolAttr isVolatile) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(coord);
  odsState.addOperands(barrier);
  odsState.addOperands(result);
  odsState.addOperands(pred);
  if (cache) {
    odsState.getOrAddProperties<Properties>().cache = cache;
  }
  if (evict) {
    odsState.getOrAddProperties<Properties>().evict = evict;
  }
  if (isVolatile) {
    odsState.getOrAddProperties<Properties>().isVolatile = isVolatile;
  }
}

void AsyncTMACopyGlobalToLocalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value barrier, ::mlir::Value result, ::mlir::Value pred, ::mlir::triton::CacheModifierAttr cache, ::mlir::triton::EvictionPolicyAttr evict, ::mlir::BoolAttr isVolatile) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(coord);
  odsState.addOperands(barrier);
  odsState.addOperands(result);
  odsState.addOperands(pred);
  if (cache) {
    odsState.getOrAddProperties<Properties>().cache = cache;
  }
  if (evict) {
    odsState.getOrAddProperties<Properties>().evict = evict;
  }
  if (isVolatile) {
    odsState.getOrAddProperties<Properties>().isVolatile = isVolatile;
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AsyncTMACopyGlobalToLocalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value barrier, ::mlir::Value result, ::mlir::Value pred, ::mlir::triton::CacheModifier cache, ::mlir::triton::EvictionPolicy evict, bool isVolatile) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(coord);
  odsState.addOperands(barrier);
  odsState.addOperands(result);
  odsState.addOperands(pred);
  odsState.getOrAddProperties<Properties>().cache = ::mlir::triton::CacheModifierAttr::get(odsBuilder.getContext(), cache);
  odsState.getOrAddProperties<Properties>().evict = ::mlir::triton::EvictionPolicyAttr::get(odsBuilder.getContext(), evict);
  odsState.getOrAddProperties<Properties>().isVolatile = odsBuilder.getBoolAttr(isVolatile);
}

void AsyncTMACopyGlobalToLocalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value barrier, ::mlir::Value result, ::mlir::Value pred, ::mlir::triton::CacheModifier cache, ::mlir::triton::EvictionPolicy evict, bool isVolatile) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(coord);
  odsState.addOperands(barrier);
  odsState.addOperands(result);
  odsState.addOperands(pred);
  odsState.getOrAddProperties<Properties>().cache = ::mlir::triton::CacheModifierAttr::get(odsBuilder.getContext(), cache);
  odsState.getOrAddProperties<Properties>().evict = ::mlir::triton::EvictionPolicyAttr::get(odsBuilder.getContext(), evict);
  odsState.getOrAddProperties<Properties>().isVolatile = odsBuilder.getBoolAttr(isVolatile);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AsyncTMACopyGlobalToLocalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<AsyncTMACopyGlobalToLocalOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void AsyncTMACopyGlobalToLocalOp::populateDefaultProperties(::mlir::OperationName opName, Properties &properties) {
  ::mlir::Builder odsBuilder(opName.getContext());
  if (!properties.cache)
    properties.cache = ::mlir::triton::CacheModifierAttr::get(odsBuilder.getContext(), triton::CacheModifier::NONE);
  if (!properties.evict)
    properties.evict = ::mlir::triton::EvictionPolicyAttr::get(odsBuilder.getContext(), triton::EvictionPolicy::NORMAL);
  if (!properties.isVolatile)
    properties.isVolatile = odsBuilder.getBoolAttr(false);
}

::mlir::LogicalResult AsyncTMACopyGlobalToLocalOp::verifyInvariantsImpl() {
  auto tblgen_cache = getProperties().cache; (void)tblgen_cache;
  auto tblgen_evict = getProperties().evict; (void)tblgen_evict;
  auto tblgen_isVolatile = getProperties().isVolatile; (void)tblgen_isVolatile;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps0(*this, tblgen_cache, "cache")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps1(*this, tblgen_evict, "evict")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps2(*this, tblgen_isVolatile, "isVolatile")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AsyncTMACopyGlobalToLocalOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AsyncTMACopyGlobalToLocalOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand desc_ptrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> desc_ptrOperands(&desc_ptrRawOperand, 1);  ::llvm::SMLoc desc_ptrOperandsLoc;
  (void)desc_ptrOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> coordOperands;
  ::llvm::SMLoc coordOperandsLoc;
  (void)coordOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand resultRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> resultOperands(&resultRawOperand, 1);  ::llvm::SMLoc resultOperandsLoc;
  (void)resultOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand barrierRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> barrierOperands(&barrierRawOperand, 1);  ::llvm::SMLoc barrierOperandsLoc;
  (void)barrierOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand predRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> predOperands(&predRawOperand, 1);  ::llvm::SMLoc predOperandsLoc;
  (void)predOperandsLoc;
  ::mlir::triton::CacheModifierAttr cacheAttr;
  ::mlir::triton::EvictionPolicyAttr evictAttr;
  ::mlir::Type desc_ptrRawType{};
  ::llvm::ArrayRef<::mlir::Type> desc_ptrTypes(&desc_ptrRawType, 1);
  ::mlir::Type barrierRawType{};
  ::llvm::ArrayRef<::mlir::Type> barrierTypes(&barrierRawType, 1);
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);

  desc_ptrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(desc_ptrRawOperand))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  coordOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(coordOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();

  resultOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(resultRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  barrierOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(barrierRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  predOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(predRawOperand))
    return ::mlir::failure();
  bool cacheModifierClause = false;
  bool evictionPolicyClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("cacheModifier"))) {

  if (cacheModifierClause) {
    return parser.emitError(parser.getNameLoc())
          << "`cacheModifier` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  cacheModifierClause = true;
  if (parser.parseEqual())
    return ::mlir::failure();

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"none","ca","cg","wb","cs","wt"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "cache", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'cache' [none, ca, cg, wb, cs, wt]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::triton::symbolizeCacheModifier(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "cache attribute specification: \"" << attrStr << '"';;

      cacheAttr = ::mlir::triton::CacheModifierAttr::get(parser.getBuilder().getContext(), *attrOptional);
        result.getOrAddProperties<AsyncTMACopyGlobalToLocalOp::Properties>().cache = cacheAttr;
    }
  }
    } else if (succeeded(parser.parseOptionalKeyword("evictionPolicy"))) {

  if (evictionPolicyClause) {
    return parser.emitError(parser.getNameLoc())
          << "`evictionPolicy` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  evictionPolicyClause = true;
  if (parser.parseEqual())
    return ::mlir::failure();

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"evict_normal","evict_first","evict_last"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "evict", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'evict' [evict_normal, evict_first, evict_last]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::triton::symbolizeEvictionPolicy(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "evict attribute specification: \"" << attrStr << '"';;

      evictAttr = ::mlir::triton::EvictionPolicyAttr::get(parser.getBuilder().getContext(), *attrOptional);
        result.getOrAddProperties<AsyncTMACopyGlobalToLocalOp::Properties>().evict = evictAttr;
    }
  }
    } else  {
    break;
  }
}
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::triton::PointerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    desc_ptrRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::triton::MemDescType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    barrierRawType = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::triton::MemDescType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawType = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(desc_ptrOperands, desc_ptrTypes, desc_ptrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(coordOperands, odsBuildableType0, coordOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(barrierOperands, barrierTypes, barrierOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(resultOperands, resultTypes, resultOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(predOperands, odsBuildableType1, predOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AsyncTMACopyGlobalToLocalOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getDescPtr();
  _odsPrinter << "[";
  _odsPrinter << getCoord();
  _odsPrinter << "]";
  _odsPrinter << ' ';
  _odsPrinter << getResult();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getBarrier();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getPred();
  if (false || (getCacheAttr() && getCacheAttr() != ::mlir::triton::CacheModifierAttr::get(::mlir::OpBuilder((*this)->getContext()).getContext(), triton::CacheModifier::NONE))) {
  _odsPrinter << ' ' << "cacheModifier";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';

  {
    auto caseValue = getCache();
    auto caseValueStr = stringifyCacheModifier(caseValue);
    _odsPrinter << caseValueStr;
  }
  }
  if (false || (getEvictAttr() && getEvictAttr() != ::mlir::triton::EvictionPolicyAttr::get(::mlir::OpBuilder((*this)->getContext()).getContext(), triton::EvictionPolicy::NORMAL))) {
  _odsPrinter << ' ' << "evictionPolicy";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';

  {
    auto caseValue = getEvict();
    auto caseValueStr = stringifyEvictionPolicy(caseValue);
    _odsPrinter << caseValueStr;
  }
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("cache");
  elidedAttrs.push_back("evict");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getCacheAttr();
     if(attr && (attr == ::mlir::triton::CacheModifierAttr::get(odsBuilder.getContext(), triton::CacheModifier::NONE)))
       elidedAttrs.push_back("cache");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getEvictAttr();
     if(attr && (attr == ::mlir::triton::EvictionPolicyAttr::get(odsBuilder.getContext(), triton::EvictionPolicy::NORMAL)))
       elidedAttrs.push_back("evict");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getIsVolatileAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("isVolatile");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getDescPtr().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::triton::PointerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getBarrier().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::triton::MemDescType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::triton::MemDescType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AsyncTMACopyLocalToGlobalOpGenericAdaptorBase::AsyncTMACopyLocalToGlobalOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("triton_nvidia_gpu.async_tma_copy_local_to_global", odsAttrs.getContext());
}

AsyncTMACopyLocalToGlobalOpGenericAdaptorBase::AsyncTMACopyLocalToGlobalOpGenericAdaptorBase(AsyncTMACopyLocalToGlobalOp op) : AsyncTMACopyLocalToGlobalOpGenericAdaptorBase(op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

std::pair<unsigned, unsigned> AsyncTMACopyLocalToGlobalOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

} // namespace detail
AsyncTMACopyLocalToGlobalOpAdaptor::AsyncTMACopyLocalToGlobalOpAdaptor(AsyncTMACopyLocalToGlobalOp op) : AsyncTMACopyLocalToGlobalOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult AsyncTMACopyLocalToGlobalOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AsyncTMACopyLocalToGlobalOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange AsyncTMACopyLocalToGlobalOp::getCoordMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

void AsyncTMACopyLocalToGlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value src) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(coord);
  odsState.addOperands(src);
}

void AsyncTMACopyLocalToGlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value src) {
  odsState.addOperands(desc_ptr);
  odsState.addOperands(coord);
  odsState.addOperands(src);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AsyncTMACopyLocalToGlobalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AsyncTMACopyLocalToGlobalOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AsyncTMACopyLocalToGlobalOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AsyncTMACopyLocalToGlobalOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand desc_ptrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> desc_ptrOperands(&desc_ptrRawOperand, 1);  ::llvm::SMLoc desc_ptrOperandsLoc;
  (void)desc_ptrOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> coordOperands;
  ::llvm::SMLoc coordOperandsLoc;
  (void)coordOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(&srcRawOperand, 1);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::Type desc_ptrRawType{};
  ::llvm::ArrayRef<::mlir::Type> desc_ptrTypes(&desc_ptrRawType, 1);
  ::mlir::Type srcRawType{};
  ::llvm::ArrayRef<::mlir::Type> srcTypes(&srcRawType, 1);

  desc_ptrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(desc_ptrRawOperand))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  coordOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(coordOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::triton::PointerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    desc_ptrRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::triton::MemDescType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    srcRawType = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperands(desc_ptrOperands, desc_ptrTypes, desc_ptrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(coordOperands, odsBuildableType0, coordOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AsyncTMACopyLocalToGlobalOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getDescPtr();
  _odsPrinter << "[";
  _odsPrinter << getCoord();
  _odsPrinter << "]";
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getDescPtr().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::triton::PointerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getSrc().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::triton::MemDescType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::BarrierExpectOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
BarrierExpectOpGenericAdaptorBase::BarrierExpectOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("triton_nvidia_gpu.barrier_expect", odsAttrs.getContext());
}

BarrierExpectOpGenericAdaptorBase::BarrierExpectOpGenericAdaptorBase(BarrierExpectOp op) : BarrierExpectOpGenericAdaptorBase(op->getDiscardableAttrDictionary(), op.getProperties(), op->getRegions()) {}

uint32_t BarrierExpectOpGenericAdaptorBase::getSize() {
  auto attr = getSizeAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
BarrierExpectOpAdaptor::BarrierExpectOpAdaptor(BarrierExpectOp op) : BarrierExpectOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult BarrierExpectOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_size = getProperties().size; (void)tblgen_size;
  if (!tblgen_size) return emitError(loc, "'triton_nvidia_gpu.barrier_expect' op ""requires attribute 'size'");

  if (tblgen_size && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_size))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_size).getType().isSignlessInteger(32)))))
    return emitError(loc, "'triton_nvidia_gpu.barrier_expect' op ""attribute 'size' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

::mlir::LogicalResult BarrierExpectOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.size;
       auto attr = dict.get("size");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `size` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute BarrierExpectOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.size;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("size",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code BarrierExpectOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.size.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> BarrierExpectOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "size")
      return prop.size;
  return std::nullopt;
}

void BarrierExpectOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "size") {
       prop.size = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.size)>>(value);
       return;
    }
}

void BarrierExpectOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.size) attrs.append("size", prop.size);
}

::mlir::LogicalResult BarrierExpectOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getSizeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(attr, "size", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult BarrierExpectOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.size)))
    return ::mlir::failure();
  return ::mlir::success();
}

void BarrierExpectOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.size);
}

uint32_t BarrierExpectOp::getSize() {
  auto attr = getSizeAttr();
  return attr.getValue().getZExtValue();
}

void BarrierExpectOp::setSize(uint32_t attrValue) {
  getProperties().size = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void BarrierExpectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, ::mlir::IntegerAttr size, ::mlir::Value pred) {
  odsState.addOperands(alloc);
  odsState.addOperands(pred);
  odsState.getOrAddProperties<Properties>().size = size;
}

void BarrierExpectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, ::mlir::IntegerAttr size, ::mlir::Value pred) {
  odsState.addOperands(alloc);
  odsState.addOperands(pred);
  odsState.getOrAddProperties<Properties>().size = size;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BarrierExpectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, uint32_t size, ::mlir::Value pred) {
  odsState.addOperands(alloc);
  odsState.addOperands(pred);
  odsState.getOrAddProperties<Properties>().size = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), size);
}

void BarrierExpectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, uint32_t size, ::mlir::Value pred) {
  odsState.addOperands(alloc);
  odsState.addOperands(pred);
  odsState.getOrAddProperties<Properties>().size = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), size);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BarrierExpectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<BarrierExpectOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult BarrierExpectOp::verifyInvariantsImpl() {
  auto tblgen_size = getProperties().size; (void)tblgen_size;
  if (!tblgen_size) return emitOpError("requires attribute 'size'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(*this, tblgen_size, "size")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BarrierExpectOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult BarrierExpectOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand allocRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> allocOperands(&allocRawOperand, 1);  ::llvm::SMLoc allocOperandsLoc;
  (void)allocOperandsLoc;
  ::mlir::IntegerAttr sizeAttr;
  ::mlir::OpAsmParser::UnresolvedOperand predRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> predOperands(&predRawOperand, 1);  ::llvm::SMLoc predOperandsLoc;
  (void)predOperandsLoc;
  ::mlir::Type allocRawType{};
  ::llvm::ArrayRef<::mlir::Type> allocTypes(&allocRawType, 1);

  allocOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(allocRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(sizeAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (sizeAttr) result.getOrAddProperties<BarrierExpectOp::Properties>().size = sizeAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseComma())
    return ::mlir::failure();

  predOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(predRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::triton::MemDescType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    allocRawType = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(allocOperands, allocTypes, allocOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(predOperands, odsBuildableType0, predOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BarrierExpectOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAlloc();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getSizeAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("size");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getPred();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getAlloc().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::triton::MemDescType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::BarrierExpectOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::ClusterArriveOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ClusterArriveOpGenericAdaptorBase::ClusterArriveOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("triton_nvidia_gpu.cluster_arrive", odsAttrs.getContext());
}

ClusterArriveOpGenericAdaptorBase::ClusterArriveOpGenericAdaptorBase(ClusterArriveOp op) : ClusterArriveOpGenericAdaptorBase(op->getDiscardableAttrDictionary(), op.getProperties(), op->getRegions()) {}

bool ClusterArriveOpGenericAdaptorBase::getRelaxed() {
  auto attr = getRelaxedAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ClusterArriveOpAdaptor::ClusterArriveOpAdaptor(ClusterArriveOp op) : ClusterArriveOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult ClusterArriveOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_relaxed = getProperties().relaxed; (void)tblgen_relaxed;
  if (!tblgen_relaxed) return emitError(loc, "'triton_nvidia_gpu.cluster_arrive' op ""requires attribute 'relaxed'");

  if (tblgen_relaxed && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_relaxed))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_relaxed).getType().isSignlessInteger(1)))))
    return emitError(loc, "'triton_nvidia_gpu.cluster_arrive' op ""attribute 'relaxed' failed to satisfy constraint: 1-bit signless integer attribute");
  return ::mlir::success();
}

::mlir::LogicalResult ClusterArriveOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.relaxed;
       auto attr = dict.get("relaxed");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `relaxed` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ClusterArriveOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.relaxed;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("relaxed",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ClusterArriveOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.relaxed.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ClusterArriveOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "relaxed")
      return prop.relaxed;
  return std::nullopt;
}

void ClusterArriveOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "relaxed") {
       prop.relaxed = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.relaxed)>>(value);
       return;
    }
}

void ClusterArriveOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.relaxed) attrs.append("relaxed", prop.relaxed);
}

::mlir::LogicalResult ClusterArriveOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getRelaxedAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps4(attr, "relaxed", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult ClusterArriveOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.relaxed)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ClusterArriveOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.relaxed);
}

bool ClusterArriveOp::getRelaxed() {
  auto attr = getRelaxedAttr();
  return attr.getValue().getZExtValue();
}

void ClusterArriveOp::setRelaxed(bool attrValue) {
  getProperties().relaxed = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(1), attrValue);
}

void ClusterArriveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr relaxed) {
  odsState.getOrAddProperties<Properties>().relaxed = relaxed;
}

void ClusterArriveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr relaxed) {
  odsState.getOrAddProperties<Properties>().relaxed = relaxed;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ClusterArriveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool relaxed) {
  odsState.getOrAddProperties<Properties>().relaxed = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(1), relaxed);
}

void ClusterArriveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, bool relaxed) {
  odsState.getOrAddProperties<Properties>().relaxed = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(1), relaxed);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ClusterArriveOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ClusterArriveOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult ClusterArriveOp::verifyInvariantsImpl() {
  auto tblgen_relaxed = getProperties().relaxed; (void)tblgen_relaxed;
  if (!tblgen_relaxed) return emitOpError("requires attribute 'relaxed'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps4(*this, tblgen_relaxed, "relaxed")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult ClusterArriveOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ClusterArriveOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void ClusterArriveOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::ClusterArriveOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::ClusterWaitOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ClusterWaitOpGenericAdaptorBase::ClusterWaitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("triton_nvidia_gpu.cluster_wait", odsAttrs.getContext());
}

ClusterWaitOpGenericAdaptorBase::ClusterWaitOpGenericAdaptorBase(ClusterWaitOp op) : ClusterWaitOpGenericAdaptorBase(op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

} // namespace detail
ClusterWaitOpAdaptor::ClusterWaitOpAdaptor(ClusterWaitOp op) : ClusterWaitOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult ClusterWaitOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ClusterWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void ClusterWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ClusterWaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ClusterWaitOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult ClusterWaitOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ClusterWaitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void ClusterWaitOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::ClusterWaitOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::DotAsyncOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DotAsyncOpGenericAdaptorBase::DotAsyncOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("triton_nvidia_gpu.dot_async", odsAttrs.getContext());
}

DotAsyncOpGenericAdaptorBase::DotAsyncOpGenericAdaptorBase(DotAsyncOp op) : DotAsyncOpGenericAdaptorBase(op->getDiscardableAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::triton::InputPrecision DotAsyncOpGenericAdaptorBase::getInputPrecision() {
  auto attr = getInputPrecisionAttr();
  return attr.getValue();
}

uint32_t DotAsyncOpGenericAdaptorBase::getMaxNumImpreciseAcc() {
  auto attr = getMaxNumImpreciseAccAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
DotAsyncOpAdaptor::DotAsyncOpAdaptor(DotAsyncOp op) : DotAsyncOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult DotAsyncOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_inputPrecision = getProperties().inputPrecision; (void)tblgen_inputPrecision;
  if (!tblgen_inputPrecision) return emitError(loc, "'triton_nvidia_gpu.dot_async' op ""requires attribute 'inputPrecision'");
  auto tblgen_maxNumImpreciseAcc = getProperties().maxNumImpreciseAcc; (void)tblgen_maxNumImpreciseAcc;
  if (!tblgen_maxNumImpreciseAcc) return emitError(loc, "'triton_nvidia_gpu.dot_async' op ""requires attribute 'maxNumImpreciseAcc'");

  if (tblgen_inputPrecision && !((::llvm::isa<::mlir::triton::InputPrecisionAttr>(tblgen_inputPrecision))))
    return emitError(loc, "'triton_nvidia_gpu.dot_async' op ""attribute 'inputPrecision' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1, 2");

  if (tblgen_maxNumImpreciseAcc && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_maxNumImpreciseAcc))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_maxNumImpreciseAcc).getType().isSignlessInteger(32)))))
    return emitError(loc, "'triton_nvidia_gpu.dot_async' op ""attribute 'maxNumImpreciseAcc' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

::mlir::LogicalResult DotAsyncOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.inputPrecision;
       auto attr = dict.get("inputPrecision");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `inputPrecision` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.maxNumImpreciseAcc;
       auto attr = dict.get("maxNumImpreciseAcc");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `maxNumImpreciseAcc` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute DotAsyncOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.inputPrecision;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("inputPrecision",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.maxNumImpreciseAcc;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("maxNumImpreciseAcc",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code DotAsyncOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.inputPrecision.getAsOpaquePointer()), 
    llvm::hash_value(prop.maxNumImpreciseAcc.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> DotAsyncOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "inputPrecision")
      return prop.inputPrecision;

    if (name == "maxNumImpreciseAcc")
      return prop.maxNumImpreciseAcc;
  return std::nullopt;
}

void DotAsyncOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "inputPrecision") {
       prop.inputPrecision = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.inputPrecision)>>(value);
       return;
    }

    if (name == "maxNumImpreciseAcc") {
       prop.maxNumImpreciseAcc = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.maxNumImpreciseAcc)>>(value);
       return;
    }
}

void DotAsyncOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.inputPrecision) attrs.append("inputPrecision", prop.inputPrecision);

    if (prop.maxNumImpreciseAcc) attrs.append("maxNumImpreciseAcc", prop.maxNumImpreciseAcc);
}

::mlir::LogicalResult DotAsyncOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getInputPrecisionAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps5(attr, "inputPrecision", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getMaxNumImpreciseAccAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(attr, "maxNumImpreciseAcc", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult DotAsyncOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.inputPrecision)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.maxNumImpreciseAcc)))
    return ::mlir::failure();
  return ::mlir::success();
}

void DotAsyncOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.inputPrecision);
  writer.writeAttribute(prop.maxNumImpreciseAcc);
}

::mlir::triton::InputPrecision DotAsyncOp::getInputPrecision() {
  auto attr = getInputPrecisionAttr();
  return attr.getValue();
}

uint32_t DotAsyncOp::getMaxNumImpreciseAcc() {
  auto attr = getMaxNumImpreciseAccAttr();
  return attr.getValue().getZExtValue();
}

void DotAsyncOp::setInputPrecision(::mlir::triton::InputPrecision attrValue) {
  getProperties().inputPrecision = ::mlir::triton::InputPrecisionAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void DotAsyncOp::setMaxNumImpreciseAcc(uint32_t attrValue) {
  getProperties().maxNumImpreciseAcc = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void DotAsyncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type d, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::triton::InputPrecisionAttr inputPrecision, ::mlir::IntegerAttr maxNumImpreciseAcc) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  odsState.getOrAddProperties<Properties>().inputPrecision = inputPrecision;
  odsState.getOrAddProperties<Properties>().maxNumImpreciseAcc = maxNumImpreciseAcc;
  odsState.addTypes(d);
}

void DotAsyncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::triton::InputPrecisionAttr inputPrecision, ::mlir::IntegerAttr maxNumImpreciseAcc) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  odsState.getOrAddProperties<Properties>().inputPrecision = inputPrecision;
  odsState.getOrAddProperties<Properties>().maxNumImpreciseAcc = maxNumImpreciseAcc;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(DotAsyncOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void DotAsyncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::triton::InputPrecisionAttr inputPrecision, ::mlir::IntegerAttr maxNumImpreciseAcc) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  odsState.getOrAddProperties<Properties>().inputPrecision = inputPrecision;
  odsState.getOrAddProperties<Properties>().maxNumImpreciseAcc = maxNumImpreciseAcc;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DotAsyncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type d, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::triton::InputPrecision inputPrecision, uint32_t maxNumImpreciseAcc) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  odsState.getOrAddProperties<Properties>().inputPrecision = ::mlir::triton::InputPrecisionAttr::get(odsBuilder.getContext(), inputPrecision);
  odsState.getOrAddProperties<Properties>().maxNumImpreciseAcc = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), maxNumImpreciseAcc);
  odsState.addTypes(d);
}

void DotAsyncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::triton::InputPrecision inputPrecision, uint32_t maxNumImpreciseAcc) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  odsState.getOrAddProperties<Properties>().inputPrecision = ::mlir::triton::InputPrecisionAttr::get(odsBuilder.getContext(), inputPrecision);
  odsState.getOrAddProperties<Properties>().maxNumImpreciseAcc = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), maxNumImpreciseAcc);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(DotAsyncOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void DotAsyncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::triton::InputPrecision inputPrecision, uint32_t maxNumImpreciseAcc) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  odsState.getOrAddProperties<Properties>().inputPrecision = ::mlir::triton::InputPrecisionAttr::get(odsBuilder.getContext(), inputPrecision);
  odsState.getOrAddProperties<Properties>().maxNumImpreciseAcc = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), maxNumImpreciseAcc);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DotAsyncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<DotAsyncOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void DotAsyncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<DotAsyncOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(DotAsyncOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult DotAsyncOp::verifyInvariantsImpl() {
  auto tblgen_inputPrecision = getProperties().inputPrecision; (void)tblgen_inputPrecision;
  if (!tblgen_inputPrecision) return emitOpError("requires attribute 'inputPrecision'");
  auto tblgen_maxNumImpreciseAcc = getProperties().maxNumImpreciseAcc; (void)tblgen_maxNumImpreciseAcc;
  if (!tblgen_maxNumImpreciseAcc) return emitOpError("requires attribute 'maxNumImpreciseAcc'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps5(*this, tblgen_inputPrecision, "inputPrecision")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(*this, tblgen_maxNumImpreciseAcc, "maxNumImpreciseAcc")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSResults(0).begin()).getType(), (*this->getODSOperands(2).begin()).getType()))))
    return emitOpError("failed to verify that result's type matches accumulator's type");
  return ::mlir::success();
}

::mlir::LogicalResult DotAsyncOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult DotAsyncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand aRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> aOperands(&aRawOperand, 1);  ::llvm::SMLoc aOperandsLoc;
  (void)aOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bOperands(&bRawOperand, 1);  ::llvm::SMLoc bOperandsLoc;
  (void)bOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand cRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> cOperands(&cRawOperand, 1);  ::llvm::SMLoc cOperandsLoc;
  (void)cOperandsLoc;
  ::mlir::Type aRawType{};
  ::llvm::ArrayRef<::mlir::Type> aTypes(&aRawType, 1);
  ::mlir::Type bRawType{};
  ::llvm::ArrayRef<::mlir::Type> bTypes(&bRawType, 1);
  ::mlir::Type dRawType{};
  ::llvm::ArrayRef<::mlir::Type> dTypes(&dRawType, 1);

  aOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  bOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  cOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(cRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorOrMemDesc type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    aRawType = type;
  }
  if (parser.parseStar())
    return ::mlir::failure();

  {
    ::mlir::TensorOrMemDesc type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    bRawType = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    dRawType = type;
  }
  for (::mlir::Type type : dTypes) {
    (void)type;
    if (!(((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isFloat8E4M3FNUZ())) || ((elementType.isFloat8E5M2())) || ((elementType.isFloat8E5M2FNUZ())) || ((elementType.isF16())) || ((elementType.isBF16())) || ((elementType.isF32())) || ((elementType.isF64()))) || (((elementType.isSignlessInteger(1))) || ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64)))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
      return parser.emitError(parser.getNameLoc()) << "'d' must be ranked tensor of floating-point or integer values, but got " << type;
    }
  }
  result.addTypes(dTypes);
  if (parser.resolveOperands(aOperands, aTypes, aOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bOperands, bTypes, bOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(cOperands, dTypes[0], cOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DotAsyncOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getA();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getB();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getC();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getA().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorOrMemDesc>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "*";
  _odsPrinter << ' ';
  {
    auto type = getB().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorOrMemDesc>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getD().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::RankedTensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::DotAsyncOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::DotWaitOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DotWaitOpGenericAdaptorBase::DotWaitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("triton_nvidia_gpu.dot_wait", odsAttrs.getContext());
}

DotWaitOpGenericAdaptorBase::DotWaitOpGenericAdaptorBase(DotWaitOp op) : DotWaitOpGenericAdaptorBase(op->getDiscardableAttrDictionary(), op.getProperties(), op->getRegions()) {}

std::pair<unsigned, unsigned> DotWaitOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

uint32_t DotWaitOpGenericAdaptorBase::getPendings() {
  auto attr = getPendingsAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
DotWaitOpAdaptor::DotWaitOpAdaptor(DotWaitOp op) : DotWaitOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult DotWaitOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_pendings = getProperties().pendings; (void)tblgen_pendings;
  if (!tblgen_pendings) return emitError(loc, "'triton_nvidia_gpu.dot_wait' op ""requires attribute 'pendings'");

  if (tblgen_pendings && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_pendings))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_pendings).getType().isSignlessInteger(32)))))
    return emitError(loc, "'triton_nvidia_gpu.dot_wait' op ""attribute 'pendings' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DotWaitOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange DotWaitOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> DotWaitOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult DotWaitOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.pendings;
       auto attr = dict.get("pendings");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `pendings` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute DotWaitOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.pendings;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("pendings",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code DotWaitOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.pendings.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> DotWaitOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "pendings")
      return prop.pendings;
  return std::nullopt;
}

void DotWaitOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "pendings") {
       prop.pendings = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.pendings)>>(value);
       return;
    }
}

void DotWaitOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.pendings) attrs.append("pendings", prop.pendings);
}

::mlir::LogicalResult DotWaitOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getPendingsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(attr, "pendings", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult DotWaitOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.pendings)))
    return ::mlir::failure();
  return ::mlir::success();
}

void DotWaitOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.pendings);
}

uint32_t DotWaitOp::getPendings() {
  auto attr = getPendingsAttr();
  return attr.getValue().getZExtValue();
}

void DotWaitOp::setPendings(uint32_t attrValue) {
  getProperties().pendings = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void DotWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::ValueRange inputs, ::mlir::IntegerAttr pendings) {
  odsState.addOperands(inputs);
  odsState.getOrAddProperties<Properties>().pendings = pendings;
  odsState.addTypes(outputs);
}

void DotWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::IntegerAttr pendings) {
  odsState.addOperands(inputs);
  odsState.getOrAddProperties<Properties>().pendings = pendings;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(DotWaitOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void DotWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::ValueRange inputs, uint32_t pendings) {
  odsState.addOperands(inputs);
  odsState.getOrAddProperties<Properties>().pendings = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), pendings);
  odsState.addTypes(outputs);
}

void DotWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, uint32_t pendings) {
  odsState.addOperands(inputs);
  odsState.getOrAddProperties<Properties>().pendings = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), pendings);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(DotWaitOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void DotWaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<DotWaitOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void DotWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<DotWaitOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(DotWaitOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult DotWaitOp::verifyInvariantsImpl() {
  auto tblgen_pendings = getProperties().pendings; (void)tblgen_pendings;
  if (!tblgen_pendings) return emitOpError("requires attribute 'pendings'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(*this, tblgen_pendings, "pendings")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!(((this->getODSOperands(0).getType()) == (this->getODSResults(0).getType()) && (this->getODSResults(0).getType()) == (this->getODSOperands(0).getType()))))
    return emitOpError("failed to verify that all of {inputs, outputs} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult DotWaitOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult DotWaitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  ::llvm::SMLoc inputsOperandsLoc;
  (void)inputsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> inputsTypes;

  inputsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputsOperands))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(inputsTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc, result.operands))
    return ::mlir::failure();

  ::llvm::SmallVector<::mlir::Type> inferredReturnTypes;
  if (::mlir::failed(DotWaitOp::inferReturnTypes(parser.getContext(),
      result.location, result.operands,
      result.attributes.getDictionary(parser.getContext()),
      result.getRawProperties(),
      result.regions, inferredReturnTypes)))
    return ::mlir::failure();
  result.addTypes(inferredReturnTypes);
  return ::mlir::success();
}

void DotWaitOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInputs();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getInputs().getTypes();
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::DotWaitOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::FenceAsyncSharedOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FenceAsyncSharedOpGenericAdaptorBase::FenceAsyncSharedOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("triton_nvidia_gpu.fence_async_shared", odsAttrs.getContext());
}

FenceAsyncSharedOpGenericAdaptorBase::FenceAsyncSharedOpGenericAdaptorBase(FenceAsyncSharedOp op) : FenceAsyncSharedOpGenericAdaptorBase(op->getDiscardableAttrDictionary(), op.getProperties(), op->getRegions()) {}

bool FenceAsyncSharedOpGenericAdaptorBase::getBCluster() {
  auto attr = getBClusterAttr();
  return attr.getValue();
}

} // namespace detail
FenceAsyncSharedOpAdaptor::FenceAsyncSharedOpAdaptor(FenceAsyncSharedOp op) : FenceAsyncSharedOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult FenceAsyncSharedOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_bCluster = getProperties().bCluster; (void)tblgen_bCluster;
  if (!tblgen_bCluster) return emitError(loc, "'triton_nvidia_gpu.fence_async_shared' op ""requires attribute 'bCluster'");

  if (tblgen_bCluster && !((::llvm::isa<::mlir::BoolAttr>(tblgen_bCluster))))
    return emitError(loc, "'triton_nvidia_gpu.fence_async_shared' op ""attribute 'bCluster' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

::mlir::LogicalResult FenceAsyncSharedOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.bCluster;
       auto attr = dict.get("bCluster");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `bCluster` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute FenceAsyncSharedOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.bCluster;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("bCluster",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code FenceAsyncSharedOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.bCluster.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> FenceAsyncSharedOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "bCluster")
      return prop.bCluster;
  return std::nullopt;
}

void FenceAsyncSharedOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "bCluster") {
       prop.bCluster = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.bCluster)>>(value);
       return;
    }
}

void FenceAsyncSharedOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.bCluster) attrs.append("bCluster", prop.bCluster);
}

::mlir::LogicalResult FenceAsyncSharedOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getBClusterAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps2(attr, "bCluster", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult FenceAsyncSharedOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.bCluster)))
    return ::mlir::failure();
  return ::mlir::success();
}

void FenceAsyncSharedOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.bCluster);
}

bool FenceAsyncSharedOp::getBCluster() {
  auto attr = getBClusterAttr();
  return attr.getValue();
}

void FenceAsyncSharedOp::setBCluster(bool attrValue) {
  getProperties().bCluster = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void FenceAsyncSharedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::BoolAttr bCluster) {
  odsState.getOrAddProperties<Properties>().bCluster = bCluster;
}

void FenceAsyncSharedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::BoolAttr bCluster) {
  odsState.getOrAddProperties<Properties>().bCluster = bCluster;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FenceAsyncSharedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool bCluster) {
  odsState.getOrAddProperties<Properties>().bCluster = odsBuilder.getBoolAttr(bCluster);
}

void FenceAsyncSharedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, bool bCluster) {
  odsState.getOrAddProperties<Properties>().bCluster = odsBuilder.getBoolAttr(bCluster);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FenceAsyncSharedOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<FenceAsyncSharedOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult FenceAsyncSharedOp::verifyInvariantsImpl() {
  auto tblgen_bCluster = getProperties().bCluster; (void)tblgen_bCluster;
  if (!tblgen_bCluster) return emitOpError("requires attribute 'bCluster'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps2(*this, tblgen_bCluster, "bCluster")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult FenceAsyncSharedOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FenceAsyncSharedOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void FenceAsyncSharedOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::FenceAsyncSharedOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::InitBarrierOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
InitBarrierOpGenericAdaptorBase::InitBarrierOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("triton_nvidia_gpu.init_barrier", odsAttrs.getContext());
}

InitBarrierOpGenericAdaptorBase::InitBarrierOpGenericAdaptorBase(InitBarrierOp op) : InitBarrierOpGenericAdaptorBase(op->getDiscardableAttrDictionary(), op.getProperties(), op->getRegions()) {}

uint32_t InitBarrierOpGenericAdaptorBase::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
InitBarrierOpAdaptor::InitBarrierOpAdaptor(InitBarrierOp op) : InitBarrierOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult InitBarrierOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_count = getProperties().count; (void)tblgen_count;
  if (!tblgen_count) return emitError(loc, "'triton_nvidia_gpu.init_barrier' op ""requires attribute 'count'");

  if (tblgen_count && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_count))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_count).getType().isSignlessInteger(32)))))
    return emitError(loc, "'triton_nvidia_gpu.init_barrier' op ""attribute 'count' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

::mlir::LogicalResult InitBarrierOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.count;
       auto attr = dict.get("count");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `count` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute InitBarrierOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.count;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("count",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code InitBarrierOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.count.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> InitBarrierOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "count")
      return prop.count;
  return std::nullopt;
}

void InitBarrierOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "count") {
       prop.count = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.count)>>(value);
       return;
    }
}

void InitBarrierOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.count) attrs.append("count", prop.count);
}

::mlir::LogicalResult InitBarrierOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getCountAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(attr, "count", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult InitBarrierOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.count)))
    return ::mlir::failure();
  return ::mlir::success();
}

void InitBarrierOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.count);
}

uint32_t InitBarrierOp::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

void InitBarrierOp::setCount(uint32_t attrValue) {
  getProperties().count = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void InitBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, ::mlir::IntegerAttr count) {
  odsState.addOperands(alloc);
  odsState.getOrAddProperties<Properties>().count = count;
}

void InitBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, ::mlir::IntegerAttr count) {
  odsState.addOperands(alloc);
  odsState.getOrAddProperties<Properties>().count = count;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InitBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, uint32_t count) {
  odsState.addOperands(alloc);
  odsState.getOrAddProperties<Properties>().count = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count);
}

void InitBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, uint32_t count) {
  odsState.addOperands(alloc);
  odsState.getOrAddProperties<Properties>().count = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InitBarrierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<InitBarrierOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult InitBarrierOp::verifyInvariantsImpl() {
  auto tblgen_count = getProperties().count; (void)tblgen_count;
  if (!tblgen_count) return emitOpError("requires attribute 'count'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(*this, tblgen_count, "count")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult InitBarrierOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult InitBarrierOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand allocRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> allocOperands(&allocRawOperand, 1);  ::llvm::SMLoc allocOperandsLoc;
  (void)allocOperandsLoc;
  ::mlir::IntegerAttr countAttr;
  ::mlir::Type allocRawType{};
  ::llvm::ArrayRef<::mlir::Type> allocTypes(&allocRawType, 1);

  allocOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(allocRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(countAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (countAttr) result.getOrAddProperties<InitBarrierOp::Properties>().count = countAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::triton::MemDescType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    allocRawType = type;
  }
  if (parser.resolveOperands(allocOperands, allocTypes, allocOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InitBarrierOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAlloc();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCountAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("count");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getAlloc().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::triton::MemDescType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::InitBarrierOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::InvalBarrierOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
InvalBarrierOpGenericAdaptorBase::InvalBarrierOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("triton_nvidia_gpu.inval_barrier", odsAttrs.getContext());
}

InvalBarrierOpGenericAdaptorBase::InvalBarrierOpGenericAdaptorBase(InvalBarrierOp op) : InvalBarrierOpGenericAdaptorBase(op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

} // namespace detail
InvalBarrierOpAdaptor::InvalBarrierOpAdaptor(InvalBarrierOp op) : InvalBarrierOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult InvalBarrierOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void InvalBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc) {
  odsState.addOperands(alloc);
}

void InvalBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc) {
  odsState.addOperands(alloc);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InvalBarrierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InvalBarrierOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult InvalBarrierOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult InvalBarrierOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand allocRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> allocOperands(&allocRawOperand, 1);  ::llvm::SMLoc allocOperandsLoc;
  (void)allocOperandsLoc;
  ::mlir::Type allocRawType{};
  ::llvm::ArrayRef<::mlir::Type> allocTypes(&allocRawType, 1);

  allocOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(allocRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::triton::MemDescType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    allocRawType = type;
  }
  if (parser.resolveOperands(allocOperands, allocTypes, allocOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InvalBarrierOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAlloc();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getAlloc().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::triton::MemDescType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::InvalBarrierOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::TMAStoreWait definitions
//===----------------------------------------------------------------------===//

namespace detail {
TMAStoreWaitGenericAdaptorBase::TMAStoreWaitGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("triton_nvidia_gpu.async_tma_store_wait", odsAttrs.getContext());
}

TMAStoreWaitGenericAdaptorBase::TMAStoreWaitGenericAdaptorBase(TMAStoreWait op) : TMAStoreWaitGenericAdaptorBase(op->getDiscardableAttrDictionary(), op.getProperties(), op->getRegions()) {}

uint32_t TMAStoreWaitGenericAdaptorBase::getPendings() {
  auto attr = getPendingsAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
TMAStoreWaitAdaptor::TMAStoreWaitAdaptor(TMAStoreWait op) : TMAStoreWaitGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult TMAStoreWaitAdaptor::verify(::mlir::Location loc) {
  auto tblgen_pendings = getProperties().pendings; (void)tblgen_pendings;
  if (!tblgen_pendings) return emitError(loc, "'triton_nvidia_gpu.async_tma_store_wait' op ""requires attribute 'pendings'");

  if (tblgen_pendings && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_pendings))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_pendings).getType().isSignlessInteger(32)))))
    return emitError(loc, "'triton_nvidia_gpu.async_tma_store_wait' op ""attribute 'pendings' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

::mlir::LogicalResult TMAStoreWait::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.pendings;
       auto attr = dict.get("pendings");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `pendings` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute TMAStoreWait::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.pendings;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("pendings",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TMAStoreWait::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.pendings.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> TMAStoreWait::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "pendings")
      return prop.pendings;
  return std::nullopt;
}

void TMAStoreWait::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "pendings") {
       prop.pendings = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.pendings)>>(value);
       return;
    }
}

void TMAStoreWait::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.pendings) attrs.append("pendings", prop.pendings);
}

::mlir::LogicalResult TMAStoreWait::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getPendingsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(attr, "pendings", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult TMAStoreWait::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.pendings)))
    return ::mlir::failure();
  return ::mlir::success();
}

void TMAStoreWait::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.pendings);
}

uint32_t TMAStoreWait::getPendings() {
  auto attr = getPendingsAttr();
  return attr.getValue().getZExtValue();
}

void TMAStoreWait::setPendings(uint32_t attrValue) {
  getProperties().pendings = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void TMAStoreWait::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr pendings) {
  odsState.getOrAddProperties<Properties>().pendings = pendings;
}

void TMAStoreWait::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr pendings) {
  odsState.getOrAddProperties<Properties>().pendings = pendings;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TMAStoreWait::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint32_t pendings) {
  odsState.getOrAddProperties<Properties>().pendings = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), pendings);
}

void TMAStoreWait::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t pendings) {
  odsState.getOrAddProperties<Properties>().pendings = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), pendings);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TMAStoreWait::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<TMAStoreWait::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult TMAStoreWait::verifyInvariantsImpl() {
  auto tblgen_pendings = getProperties().pendings; (void)tblgen_pendings;
  if (!tblgen_pendings) return emitOpError("requires attribute 'pendings'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonNvidiaGPUOps3(*this, tblgen_pendings, "pendings")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult TMAStoreWait::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TMAStoreWait::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void TMAStoreWait::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TMAStoreWait)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::WaitBarrierOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
WaitBarrierOpGenericAdaptorBase::WaitBarrierOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("triton_nvidia_gpu.wait_barrier", odsAttrs.getContext());
}

WaitBarrierOpGenericAdaptorBase::WaitBarrierOpGenericAdaptorBase(WaitBarrierOp op) : WaitBarrierOpGenericAdaptorBase(op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

} // namespace detail
WaitBarrierOpAdaptor::WaitBarrierOpAdaptor(WaitBarrierOp op) : WaitBarrierOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult WaitBarrierOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void WaitBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, ::mlir::Value phase) {
  odsState.addOperands(alloc);
  odsState.addOperands(phase);
}

void WaitBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, ::mlir::Value phase) {
  odsState.addOperands(alloc);
  odsState.addOperands(phase);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WaitBarrierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WaitBarrierOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonNvidiaGPUOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult WaitBarrierOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult WaitBarrierOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand allocRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> allocOperands(&allocRawOperand, 1);  ::llvm::SMLoc allocOperandsLoc;
  (void)allocOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand phaseRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> phaseOperands(&phaseRawOperand, 1);  ::llvm::SMLoc phaseOperandsLoc;
  (void)phaseOperandsLoc;
  ::mlir::Type allocRawType{};
  ::llvm::ArrayRef<::mlir::Type> allocTypes(&allocRawType, 1);

  allocOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(allocRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  phaseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(phaseRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::triton::MemDescType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    allocRawType = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperands(allocOperands, allocTypes, allocOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(phaseOperands, odsBuildableType0, phaseOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WaitBarrierOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAlloc();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getPhase();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getAlloc().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::triton::MemDescType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::WaitBarrierOp)


#endif  // GET_OP_CLASSES

