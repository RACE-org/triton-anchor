/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonNvidiaGPUOps.td                                                *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace triton {
namespace nvidia_gpu {
class AsyncTMACopyGlobalToLocalOp;
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace nvidia_gpu {
class AsyncTMACopyLocalToGlobalOp;
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace nvidia_gpu {
class BarrierExpectOp;
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace nvidia_gpu {
class ClusterArriveOp;
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace nvidia_gpu {
class ClusterWaitOp;
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace nvidia_gpu {
class DotAsyncOp;
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace nvidia_gpu {
class DotWaitOp;
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace nvidia_gpu {
class FenceAsyncSharedOp;
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace nvidia_gpu {
class InitBarrierOp;
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace nvidia_gpu {
class InvalBarrierOp;
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace nvidia_gpu {
class TMAStoreWait;
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
namespace mlir {
namespace triton {
namespace nvidia_gpu {
class WaitBarrierOp;
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AsyncTMACopyGlobalToLocalOpGenericAdaptorBase {
public:
  struct Properties {
    using cacheTy = ::mlir::triton::CacheModifierAttr;
    cacheTy cache;

    auto getCache() {
      auto &propStorage = this->cache;
      return ::llvm::dyn_cast_or_null<::mlir::triton::CacheModifierAttr>(propStorage);
    }
    void setCache(const ::mlir::triton::CacheModifierAttr &propValue) {
      this->cache = propValue;
    }
    using evictTy = ::mlir::triton::EvictionPolicyAttr;
    evictTy evict;

    auto getEvict() {
      auto &propStorage = this->evict;
      return ::llvm::dyn_cast_or_null<::mlir::triton::EvictionPolicyAttr>(propStorage);
    }
    void setEvict(const ::mlir::triton::EvictionPolicyAttr &propValue) {
      this->evict = propValue;
    }
    using isVolatileTy = ::mlir::BoolAttr;
    isVolatileTy isVolatile;

    auto getIsVolatile() {
      auto &propStorage = this->isVolatile;
      return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(propStorage);
    }
    void setIsVolatile(const ::mlir::BoolAttr &propValue) {
      this->isVolatile = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.cache == this->cache &&
        rhs.evict == this->evict &&
        rhs.isVolatile == this->isVolatile &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AsyncTMACopyGlobalToLocalOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {});

  AsyncTMACopyGlobalToLocalOpGenericAdaptorBase(AsyncTMACopyGlobalToLocalOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::triton::CacheModifierAttr getCacheAttr();
  ::mlir::triton::CacheModifier getCache();
  ::mlir::triton::EvictionPolicyAttr getEvictAttr();
  ::mlir::triton::EvictionPolicy getEvict();
  ::mlir::BoolAttr getIsVolatileAttr();
  bool getIsVolatile();
};
} // namespace detail
template <typename RangeT>
class AsyncTMACopyGlobalToLocalOpGenericAdaptor : public detail::AsyncTMACopyGlobalToLocalOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AsyncTMACopyGlobalToLocalOpGenericAdaptorBase;
public:
  AsyncTMACopyGlobalToLocalOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AsyncTMACopyGlobalToLocalOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AsyncTMACopyGlobalToLocalOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AsyncTMACopyGlobalToLocalOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AsyncTMACopyGlobalToLocalOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = AsyncTMACopyGlobalToLocalOp, typename = std::enable_if_t<std::is_same_v<LateInst, AsyncTMACopyGlobalToLocalOp>>>
  AsyncTMACopyGlobalToLocalOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getDescPtr() {
    return (*getODSOperands(0).begin());
  }

  RangeT getCoord() {
    return getODSOperands(1);
  }

  ValueT getBarrier() {
    return (*getODSOperands(2).begin());
  }

  ValueT getResult() {
    return (*getODSOperands(3).begin());
  }

  ValueT getPred() {
    return (*getODSOperands(4).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AsyncTMACopyGlobalToLocalOpAdaptor : public AsyncTMACopyGlobalToLocalOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AsyncTMACopyGlobalToLocalOpGenericAdaptor::AsyncTMACopyGlobalToLocalOpGenericAdaptor;
  AsyncTMACopyGlobalToLocalOpAdaptor(AsyncTMACopyGlobalToLocalOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AsyncTMACopyGlobalToLocalOp : public ::mlir::Op<AsyncTMACopyGlobalToLocalOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::VerifyTensorLayoutsTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AsyncTMACopyGlobalToLocalOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AsyncTMACopyGlobalToLocalOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("cache"), ::llvm::StringRef("evict"), ::llvm::StringRef("isVolatile")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getCacheAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getCacheAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getEvictAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getEvictAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getIsVolatileAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getIsVolatileAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_nvidia_gpu.async_tma_copy_global_to_local");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::PointerType> getDescPtr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::PointerType>>(*getODSOperands(0).begin());
  }

  ::mlir::Operation::operand_range getCoord() {
    return getODSOperands(1);
  }

  ::mlir::TypedValue<::mlir::triton::MemDescType> getBarrier() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::MemDescType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::triton::MemDescType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::MemDescType>>(*getODSOperands(3).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPred() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(4).begin());
  }

  ::mlir::OpOperand &getDescPtrMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getCoordMutable();
  ::mlir::OpOperand &getBarrierMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getResultMutable() {
    auto range = getODSOperandIndexAndLength(3);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getPredMutable() {
    auto range = getODSOperandIndexAndLength(4);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::mlir::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::mlir::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::triton::CacheModifierAttr getCacheAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::triton::CacheModifierAttr>(getProperties().cache);
  }

  ::mlir::triton::CacheModifier getCache();
  ::mlir::triton::EvictionPolicyAttr getEvictAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::triton::EvictionPolicyAttr>(getProperties().evict);
  }

  ::mlir::triton::EvictionPolicy getEvict();
  ::mlir::BoolAttr getIsVolatileAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().isVolatile);
  }

  bool getIsVolatile();
  void setCacheAttr(::mlir::triton::CacheModifierAttr attr) {
    getProperties().cache = attr;
  }

  void setCache(::mlir::triton::CacheModifier attrValue);
  void setEvictAttr(::mlir::triton::EvictionPolicyAttr attr) {
    getProperties().evict = attr;
  }

  void setEvict(::mlir::triton::EvictionPolicy attrValue);
  void setIsVolatileAttr(::mlir::BoolAttr attr) {
    getProperties().isVolatile = attr;
  }

  void setIsVolatile(bool attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value barrier, ::mlir::Value result, ::mlir::Value pred, ::mlir::triton::CacheModifierAttr cache, ::mlir::triton::EvictionPolicyAttr evict = nullptr, ::mlir::BoolAttr isVolatile = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value barrier, ::mlir::Value result, ::mlir::Value pred, ::mlir::triton::CacheModifierAttr cache, ::mlir::triton::EvictionPolicyAttr evict = nullptr, ::mlir::BoolAttr isVolatile = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value barrier, ::mlir::Value result, ::mlir::Value pred, ::mlir::triton::CacheModifier cache = triton::CacheModifier::NONE, ::mlir::triton::EvictionPolicy evict = triton::EvictionPolicy::NORMAL, bool isVolatile = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value barrier, ::mlir::Value result, ::mlir::Value pred, ::mlir::triton::CacheModifier cache = triton::CacheModifier::NONE, ::mlir::triton::EvictionPolicy evict = triton::EvictionPolicy::NORMAL, bool isVolatile = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultProperties(::mlir::OperationName opName, Properties &properties);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AsyncTMACopyLocalToGlobalOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  AsyncTMACopyLocalToGlobalOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {});

  AsyncTMACopyLocalToGlobalOpGenericAdaptorBase(AsyncTMACopyLocalToGlobalOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};
} // namespace detail
template <typename RangeT>
class AsyncTMACopyLocalToGlobalOpGenericAdaptor : public detail::AsyncTMACopyLocalToGlobalOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AsyncTMACopyLocalToGlobalOpGenericAdaptorBase;
public:
  AsyncTMACopyLocalToGlobalOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AsyncTMACopyLocalToGlobalOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AsyncTMACopyLocalToGlobalOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  template <typename LateInst = AsyncTMACopyLocalToGlobalOp, typename = std::enable_if_t<std::is_same_v<LateInst, AsyncTMACopyLocalToGlobalOp>>>
  AsyncTMACopyLocalToGlobalOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getDescPtr() {
    return (*getODSOperands(0).begin());
  }

  RangeT getCoord() {
    return getODSOperands(1);
  }

  ValueT getSrc() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AsyncTMACopyLocalToGlobalOpAdaptor : public AsyncTMACopyLocalToGlobalOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AsyncTMACopyLocalToGlobalOpGenericAdaptor::AsyncTMACopyLocalToGlobalOpGenericAdaptor;
  AsyncTMACopyLocalToGlobalOpAdaptor(AsyncTMACopyLocalToGlobalOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AsyncTMACopyLocalToGlobalOp : public ::mlir::Op<AsyncTMACopyLocalToGlobalOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::VerifyTensorLayoutsTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AsyncTMACopyLocalToGlobalOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AsyncTMACopyLocalToGlobalOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_nvidia_gpu.async_tma_copy_local_to_global");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::PointerType> getDescPtr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::PointerType>>(*getODSOperands(0).begin());
  }

  ::mlir::Operation::operand_range getCoord() {
    return getODSOperands(1);
  }

  ::mlir::TypedValue<::mlir::triton::MemDescType> getSrc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::MemDescType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getDescPtrMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getCoordMutable();
  ::mlir::OpOperand &getSrcMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value src);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc_ptr, ::mlir::ValueRange coord, ::mlir::Value src);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::BarrierExpectOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class BarrierExpectOpGenericAdaptorBase {
public:
  struct Properties {
    using sizeTy = ::mlir::IntegerAttr;
    sizeTy size;

    auto getSize() {
      auto &propStorage = this->size;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setSize(const ::mlir::IntegerAttr &propValue) {
      this->size = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.size == this->size &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  BarrierExpectOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {});

  BarrierExpectOpGenericAdaptorBase(BarrierExpectOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getSizeAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().size);
    return attr;
  }

  uint32_t getSize();
};
} // namespace detail
template <typename RangeT>
class BarrierExpectOpGenericAdaptor : public detail::BarrierExpectOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::BarrierExpectOpGenericAdaptorBase;
public:
  BarrierExpectOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  BarrierExpectOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : BarrierExpectOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  BarrierExpectOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : BarrierExpectOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = BarrierExpectOp, typename = std::enable_if_t<std::is_same_v<LateInst, BarrierExpectOp>>>
  BarrierExpectOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAlloc() {
    return (*getODSOperands(0).begin());
  }

  ValueT getPred() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class BarrierExpectOpAdaptor : public BarrierExpectOpGenericAdaptor<::mlir::ValueRange> {
public:
  using BarrierExpectOpGenericAdaptor::BarrierExpectOpGenericAdaptor;
  BarrierExpectOpAdaptor(BarrierExpectOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class BarrierExpectOp : public ::mlir::Op<BarrierExpectOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::VerifyTensorLayoutsTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BarrierExpectOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = BarrierExpectOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("size")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getSizeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getSizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_nvidia_gpu.barrier_expect");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::MemDescType> getAlloc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::MemDescType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPred() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getAllocMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getPredMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::mlir::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::mlir::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getSizeAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().size);
  }

  uint32_t getSize();
  void setSizeAttr(::mlir::IntegerAttr attr) {
    getProperties().size = attr;
  }

  void setSize(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, ::mlir::IntegerAttr size, ::mlir::Value pred);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, ::mlir::IntegerAttr size, ::mlir::Value pred);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, uint32_t size, ::mlir::Value pred);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, uint32_t size, ::mlir::Value pred);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::BarrierExpectOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::ClusterArriveOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ClusterArriveOpGenericAdaptorBase {
public:
  struct Properties {
    using relaxedTy = ::mlir::IntegerAttr;
    relaxedTy relaxed;

    auto getRelaxed() {
      auto &propStorage = this->relaxed;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setRelaxed(const ::mlir::IntegerAttr &propValue) {
      this->relaxed = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.relaxed == this->relaxed &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ClusterArriveOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {});

  ClusterArriveOpGenericAdaptorBase(ClusterArriveOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getRelaxedAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().relaxed);
    return attr;
  }

  bool getRelaxed();
};
} // namespace detail
template <typename RangeT>
class ClusterArriveOpGenericAdaptor : public detail::ClusterArriveOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ClusterArriveOpGenericAdaptorBase;
public:
  ClusterArriveOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ClusterArriveOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ClusterArriveOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ClusterArriveOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ClusterArriveOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = ClusterArriveOp, typename = std::enable_if_t<std::is_same_v<LateInst, ClusterArriveOp>>>
  ClusterArriveOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ClusterArriveOpAdaptor : public ClusterArriveOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ClusterArriveOpGenericAdaptor::ClusterArriveOpGenericAdaptor;
  ClusterArriveOpAdaptor(ClusterArriveOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ClusterArriveOp : public ::mlir::Op<ClusterArriveOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::OpTrait::VerifyTensorLayoutsTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ClusterArriveOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ClusterArriveOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("relaxed")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getRelaxedAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getRelaxedAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_nvidia_gpu.cluster_arrive");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::mlir::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::mlir::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getRelaxedAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().relaxed);
  }

  bool getRelaxed();
  void setRelaxedAttr(::mlir::IntegerAttr attr) {
    getProperties().relaxed = attr;
  }

  void setRelaxed(bool attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr relaxed);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr relaxed);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool relaxed);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, bool relaxed);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::ClusterArriveOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::ClusterWaitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ClusterWaitOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  ClusterWaitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {});

  ClusterWaitOpGenericAdaptorBase(ClusterWaitOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};
} // namespace detail
template <typename RangeT>
class ClusterWaitOpGenericAdaptor : public detail::ClusterWaitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ClusterWaitOpGenericAdaptorBase;
public:
  ClusterWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ClusterWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ClusterWaitOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  template <typename LateInst = ClusterWaitOp, typename = std::enable_if_t<std::is_same_v<LateInst, ClusterWaitOp>>>
  ClusterWaitOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ClusterWaitOpAdaptor : public ClusterWaitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ClusterWaitOpGenericAdaptor::ClusterWaitOpGenericAdaptor;
  ClusterWaitOpAdaptor(ClusterWaitOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ClusterWaitOp : public ::mlir::Op<ClusterWaitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::VerifyTensorLayoutsTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ClusterWaitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ClusterWaitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_nvidia_gpu.cluster_wait");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::ClusterWaitOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::DotAsyncOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class DotAsyncOpGenericAdaptorBase {
public:
  struct Properties {
    using inputPrecisionTy = ::mlir::triton::InputPrecisionAttr;
    inputPrecisionTy inputPrecision;

    auto getInputPrecision() {
      auto &propStorage = this->inputPrecision;
      return ::llvm::cast<::mlir::triton::InputPrecisionAttr>(propStorage);
    }
    void setInputPrecision(const ::mlir::triton::InputPrecisionAttr &propValue) {
      this->inputPrecision = propValue;
    }
    using maxNumImpreciseAccTy = ::mlir::IntegerAttr;
    maxNumImpreciseAccTy maxNumImpreciseAcc;

    auto getMaxNumImpreciseAcc() {
      auto &propStorage = this->maxNumImpreciseAcc;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setMaxNumImpreciseAcc(const ::mlir::IntegerAttr &propValue) {
      this->maxNumImpreciseAcc = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.inputPrecision == this->inputPrecision &&
        rhs.maxNumImpreciseAcc == this->maxNumImpreciseAcc &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  DotAsyncOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {});

  DotAsyncOpGenericAdaptorBase(DotAsyncOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::triton::InputPrecisionAttr getInputPrecisionAttr() {
    auto attr = ::llvm::cast<::mlir::triton::InputPrecisionAttr>(getProperties().inputPrecision);
    return attr;
  }

  ::mlir::triton::InputPrecision getInputPrecision();
  ::mlir::IntegerAttr getMaxNumImpreciseAccAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().maxNumImpreciseAcc);
    return attr;
  }

  uint32_t getMaxNumImpreciseAcc();
};
} // namespace detail
template <typename RangeT>
class DotAsyncOpGenericAdaptor : public detail::DotAsyncOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DotAsyncOpGenericAdaptorBase;
public:
  DotAsyncOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  DotAsyncOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : DotAsyncOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  DotAsyncOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : DotAsyncOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = DotAsyncOp, typename = std::enable_if_t<std::is_same_v<LateInst, DotAsyncOp>>>
  DotAsyncOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getA() {
    return (*getODSOperands(0).begin());
  }

  ValueT getB() {
    return (*getODSOperands(1).begin());
  }

  ValueT getC() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DotAsyncOpAdaptor : public DotAsyncOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DotAsyncOpGenericAdaptor::DotAsyncOpGenericAdaptor;
  DotAsyncOpAdaptor(DotAsyncOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class DotAsyncOp : public ::mlir::Op<DotAsyncOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::VerifyTensorLayoutsTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DotAsyncOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DotAsyncOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("inputPrecision"), ::llvm::StringRef("maxNumImpreciseAcc")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getInputPrecisionAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInputPrecisionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMaxNumImpreciseAccAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMaxNumImpreciseAccAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_nvidia_gpu.dot_async");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorOrMemDesc> getA() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorOrMemDesc>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorOrMemDesc> getB() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorOrMemDesc>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getC() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getAMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getCMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getD() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
  }

  static ::mlir::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::mlir::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::triton::InputPrecisionAttr getInputPrecisionAttr() {
    return ::llvm::cast<::mlir::triton::InputPrecisionAttr>(getProperties().inputPrecision);
  }

  ::mlir::triton::InputPrecision getInputPrecision();
  ::mlir::IntegerAttr getMaxNumImpreciseAccAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().maxNumImpreciseAcc);
  }

  uint32_t getMaxNumImpreciseAcc();
  void setInputPrecisionAttr(::mlir::triton::InputPrecisionAttr attr) {
    getProperties().inputPrecision = attr;
  }

  void setInputPrecision(::mlir::triton::InputPrecision attrValue);
  void setMaxNumImpreciseAccAttr(::mlir::IntegerAttr attr) {
    getProperties().maxNumImpreciseAcc = attr;
  }

  void setMaxNumImpreciseAcc(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type d, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::triton::InputPrecisionAttr inputPrecision, ::mlir::IntegerAttr maxNumImpreciseAcc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::triton::InputPrecisionAttr inputPrecision, ::mlir::IntegerAttr maxNumImpreciseAcc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::triton::InputPrecisionAttr inputPrecision, ::mlir::IntegerAttr maxNumImpreciseAcc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type d, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::triton::InputPrecision inputPrecision, uint32_t maxNumImpreciseAcc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::triton::InputPrecision inputPrecision, uint32_t maxNumImpreciseAcc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c, ::mlir::triton::InputPrecision inputPrecision, uint32_t maxNumImpreciseAcc);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::DotAsyncOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::DotWaitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class DotWaitOpGenericAdaptorBase {
public:
  struct Properties {
    using pendingsTy = ::mlir::IntegerAttr;
    pendingsTy pendings;

    auto getPendings() {
      auto &propStorage = this->pendings;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setPendings(const ::mlir::IntegerAttr &propValue) {
      this->pendings = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.pendings == this->pendings &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  DotWaitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {});

  DotWaitOpGenericAdaptorBase(DotWaitOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getPendingsAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().pendings);
    return attr;
  }

  uint32_t getPendings();
};
} // namespace detail
template <typename RangeT>
class DotWaitOpGenericAdaptor : public detail::DotWaitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DotWaitOpGenericAdaptorBase;
public:
  DotWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  DotWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : DotWaitOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  DotWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : DotWaitOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = DotWaitOp, typename = std::enable_if_t<std::is_same_v<LateInst, DotWaitOp>>>
  DotWaitOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getInputs() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DotWaitOpAdaptor : public DotWaitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DotWaitOpGenericAdaptor::DotWaitOpGenericAdaptor;
  DotWaitOpAdaptor(DotWaitOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class DotWaitOp : public ::mlir::Op<DotWaitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::OpTrait::VerifyTensorLayoutsTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DotWaitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DotWaitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("pendings")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getPendingsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getPendingsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_nvidia_gpu.dot_wait");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::operand_range getInputs() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getInputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::result_range getOutputs() {
    return getODSResults(0);
  }

  static ::mlir::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::mlir::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getPendingsAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().pendings);
  }

  uint32_t getPendings();
  void setPendingsAttr(::mlir::IntegerAttr attr) {
    getProperties().pendings = attr;
  }

  void setPendings(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::ValueRange inputs, ::mlir::IntegerAttr pendings);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::IntegerAttr pendings);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::ValueRange inputs, uint32_t pendings);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, uint32_t pendings);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::DotWaitOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::FenceAsyncSharedOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class FenceAsyncSharedOpGenericAdaptorBase {
public:
  struct Properties {
    using bClusterTy = ::mlir::BoolAttr;
    bClusterTy bCluster;

    auto getBCluster() {
      auto &propStorage = this->bCluster;
      return ::llvm::cast<::mlir::BoolAttr>(propStorage);
    }
    void setBCluster(const ::mlir::BoolAttr &propValue) {
      this->bCluster = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.bCluster == this->bCluster &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  FenceAsyncSharedOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {});

  FenceAsyncSharedOpGenericAdaptorBase(FenceAsyncSharedOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::BoolAttr getBClusterAttr() {
    auto attr = ::llvm::cast<::mlir::BoolAttr>(getProperties().bCluster);
    return attr;
  }

  bool getBCluster();
};
} // namespace detail
template <typename RangeT>
class FenceAsyncSharedOpGenericAdaptor : public detail::FenceAsyncSharedOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::FenceAsyncSharedOpGenericAdaptorBase;
public:
  FenceAsyncSharedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  FenceAsyncSharedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : FenceAsyncSharedOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  FenceAsyncSharedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : FenceAsyncSharedOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = FenceAsyncSharedOp, typename = std::enable_if_t<std::is_same_v<LateInst, FenceAsyncSharedOp>>>
  FenceAsyncSharedOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class FenceAsyncSharedOpAdaptor : public FenceAsyncSharedOpGenericAdaptor<::mlir::ValueRange> {
public:
  using FenceAsyncSharedOpGenericAdaptor::FenceAsyncSharedOpGenericAdaptor;
  FenceAsyncSharedOpAdaptor(FenceAsyncSharedOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class FenceAsyncSharedOp : public ::mlir::Op<FenceAsyncSharedOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::OpTrait::VerifyTensorLayoutsTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FenceAsyncSharedOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = FenceAsyncSharedOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("bCluster")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getBClusterAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBClusterAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_nvidia_gpu.fence_async_shared");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::mlir::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::mlir::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::BoolAttr getBClusterAttr() {
    return ::llvm::cast<::mlir::BoolAttr>(getProperties().bCluster);
  }

  bool getBCluster();
  void setBClusterAttr(::mlir::BoolAttr attr) {
    getProperties().bCluster = attr;
  }

  void setBCluster(bool attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::BoolAttr bCluster);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::BoolAttr bCluster);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, bool bCluster);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, bool bCluster);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  static bool isSupported(int computeCapability) {
    return computeCapability >= 90;
  }
};
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::FenceAsyncSharedOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::InitBarrierOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class InitBarrierOpGenericAdaptorBase {
public:
  struct Properties {
    using countTy = ::mlir::IntegerAttr;
    countTy count;

    auto getCount() {
      auto &propStorage = this->count;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setCount(const ::mlir::IntegerAttr &propValue) {
      this->count = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.count == this->count &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  InitBarrierOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {});

  InitBarrierOpGenericAdaptorBase(InitBarrierOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getCountAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().count);
    return attr;
  }

  uint32_t getCount();
};
} // namespace detail
template <typename RangeT>
class InitBarrierOpGenericAdaptor : public detail::InitBarrierOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::InitBarrierOpGenericAdaptorBase;
public:
  InitBarrierOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  InitBarrierOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : InitBarrierOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  InitBarrierOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : InitBarrierOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = InitBarrierOp, typename = std::enable_if_t<std::is_same_v<LateInst, InitBarrierOp>>>
  InitBarrierOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAlloc() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class InitBarrierOpAdaptor : public InitBarrierOpGenericAdaptor<::mlir::ValueRange> {
public:
  using InitBarrierOpGenericAdaptor::InitBarrierOpGenericAdaptor;
  InitBarrierOpAdaptor(InitBarrierOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class InitBarrierOp : public ::mlir::Op<InitBarrierOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::VerifyTensorLayoutsTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InitBarrierOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = InitBarrierOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("count")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getCountAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_nvidia_gpu.init_barrier");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::MemDescType> getAlloc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::MemDescType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getAllocMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::mlir::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::mlir::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getCountAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().count);
  }

  uint32_t getCount();
  void setCountAttr(::mlir::IntegerAttr attr) {
    getProperties().count = attr;
  }

  void setCount(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, ::mlir::IntegerAttr count);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, ::mlir::IntegerAttr count);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, uint32_t count);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, uint32_t count);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::InitBarrierOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::InvalBarrierOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class InvalBarrierOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  InvalBarrierOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {});

  InvalBarrierOpGenericAdaptorBase(InvalBarrierOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};
} // namespace detail
template <typename RangeT>
class InvalBarrierOpGenericAdaptor : public detail::InvalBarrierOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::InvalBarrierOpGenericAdaptorBase;
public:
  InvalBarrierOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  InvalBarrierOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : InvalBarrierOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  template <typename LateInst = InvalBarrierOp, typename = std::enable_if_t<std::is_same_v<LateInst, InvalBarrierOp>>>
  InvalBarrierOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAlloc() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class InvalBarrierOpAdaptor : public InvalBarrierOpGenericAdaptor<::mlir::ValueRange> {
public:
  using InvalBarrierOpGenericAdaptor::InvalBarrierOpGenericAdaptor;
  InvalBarrierOpAdaptor(InvalBarrierOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class InvalBarrierOp : public ::mlir::Op<InvalBarrierOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::VerifyTensorLayoutsTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InvalBarrierOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = InvalBarrierOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_nvidia_gpu.inval_barrier");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::MemDescType> getAlloc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::MemDescType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getAllocMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::InvalBarrierOp)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::TMAStoreWait declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TMAStoreWaitGenericAdaptorBase {
public:
  struct Properties {
    using pendingsTy = ::mlir::IntegerAttr;
    pendingsTy pendings;

    auto getPendings() {
      auto &propStorage = this->pendings;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setPendings(const ::mlir::IntegerAttr &propValue) {
      this->pendings = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.pendings == this->pendings &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  TMAStoreWaitGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {});

  TMAStoreWaitGenericAdaptorBase(TMAStoreWait op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getPendingsAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().pendings);
    return attr;
  }

  uint32_t getPendings();
};
} // namespace detail
template <typename RangeT>
class TMAStoreWaitGenericAdaptor : public detail::TMAStoreWaitGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TMAStoreWaitGenericAdaptorBase;
public:
  TMAStoreWaitGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TMAStoreWaitGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TMAStoreWaitGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  TMAStoreWaitGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : TMAStoreWaitGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = TMAStoreWait, typename = std::enable_if_t<std::is_same_v<LateInst, TMAStoreWait>>>
  TMAStoreWaitGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TMAStoreWaitAdaptor : public TMAStoreWaitGenericAdaptor<::mlir::ValueRange> {
public:
  using TMAStoreWaitGenericAdaptor::TMAStoreWaitGenericAdaptor;
  TMAStoreWaitAdaptor(TMAStoreWait op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class TMAStoreWait : public ::mlir::Op<TMAStoreWait, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::OpTrait::VerifyTensorLayoutsTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TMAStoreWaitAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TMAStoreWaitGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("pendings")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getPendingsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getPendingsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_nvidia_gpu.async_tma_store_wait");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::mlir::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::mlir::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getPendingsAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().pendings);
  }

  uint32_t getPendings();
  void setPendingsAttr(::mlir::IntegerAttr attr) {
    getProperties().pendings = attr;
  }

  void setPendings(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr pendings);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr pendings);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint32_t pendings);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t pendings);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TMAStoreWait)

namespace mlir {
namespace triton {
namespace nvidia_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::triton::nvidia_gpu::WaitBarrierOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class WaitBarrierOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  WaitBarrierOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {});

  WaitBarrierOpGenericAdaptorBase(WaitBarrierOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};
} // namespace detail
template <typename RangeT>
class WaitBarrierOpGenericAdaptor : public detail::WaitBarrierOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WaitBarrierOpGenericAdaptorBase;
public:
  WaitBarrierOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  WaitBarrierOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : WaitBarrierOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  template <typename LateInst = WaitBarrierOp, typename = std::enable_if_t<std::is_same_v<LateInst, WaitBarrierOp>>>
  WaitBarrierOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAlloc() {
    return (*getODSOperands(0).begin());
  }

  ValueT getPhase() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class WaitBarrierOpAdaptor : public WaitBarrierOpGenericAdaptor<::mlir::ValueRange> {
public:
  using WaitBarrierOpGenericAdaptor::WaitBarrierOpGenericAdaptor;
  WaitBarrierOpAdaptor(WaitBarrierOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class WaitBarrierOp : public ::mlir::Op<WaitBarrierOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::VerifyTensorLayoutsTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WaitBarrierOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WaitBarrierOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_nvidia_gpu.wait_barrier");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::MemDescType> getAlloc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::MemDescType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPhase() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getAllocMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getPhaseMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value alloc, ::mlir::Value phase);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value alloc, ::mlir::Value phase);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::WaitBarrierOp)


#endif  // GET_OP_CLASSES

