#include "ConvertLayoutOpToSPIRV.h"
#include "Utility.h"

using ::mlir::spirv::delinearize;
using ::mlir::spirv::getSharedMemoryObjectFromStruct;
using ::mlir::spirv::getStridesFromShapeAndOrder;
using ::mlir::spirv::linearize;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::isaDistributedLayout;
using ::mlir::triton::gpu::SharedEncodingAttr;

// Forward declarations
namespace SharedToDotOperandFMA {
Value convertLayout(int opIdx, Value B, Value llB, BlockedEncodingAttr dLayout,
                    Value thread, Location loc,
                    TritonGPUToSPIRVTypeConverter *typeConverter,
                    ConversionPatternRewriter &rewriter);
}

struct LocalLoadOpConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::gpu::LocalLoadOp> {
public:
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::gpu::LocalLoadOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    // TODO: do we need to check if src is shared ?
    if (isa<SharedEncodingAttr>(srcLayout) && isaDistributedLayout(dstLayout)) {
      // currently, no such case, tell me if you encounter this case.
      assert(false && "Unsupported LocalLoadOp src operand found");
      // return lowerSharedToDistributed(op, adaptor, rewriter);
    }
    if (isa<DotOperandEncodingAttr>(dstLayout) &&
        isa<BlockedEncodingAttr>(
            cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
      return lowerSharedToDotOperand(op, adaptor, rewriter);
    }
    return failure();
  }

private:
  // shared -> mma_operand
  LogicalResult
  lowerSharedToDotOperand(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto dstTensorTy = cast<RankedTensorType>(dst.getType());
    auto srcTensorTy = cast<MemDescType>(src.getType());
    auto dotOperandLayout =
        cast<DotOperandEncodingAttr>(dstTensorTy.getEncoding());
    auto sharedLayout = cast<SharedEncodingAttr>(srcTensorTy.getEncoding());

    Value res;
    if (auto blockedLayout = dyn_cast_or_null<BlockedEncodingAttr>(
            dotOperandLayout.getParent())) {
      auto dotOpLayout =
          cast<DotOperandEncodingAttr>(dstTensorTy.getEncoding());
      auto thread = getThreadId(rewriter, loc);
      res = SharedToDotOperandFMA::convertLayout(
          dotOpLayout.getOpIdx(), src, adaptor.getSrc(), blockedLayout, thread,
          loc, getTypeConverter(), rewriter);
    } else {
      assert(false && "Unsupported dot operand layout found");
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertLayoutOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::gpu::ConvertLayoutOp> {
public:
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::gpu::ConvertLayoutOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<RankedTensorType>(dst.getType());
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isaDistributedLayout(srcLayout) && isa<SharedEncodingAttr>(dstLayout)) {
      return lowerDistributedToShared(op, adaptor, rewriter);
    }
    // if (isa<SharedEncodingAttr>(srcLayout) &&
    //     isa<DotOperandEncodingAttr>(dstLayout)) {
    //   return lowerSharedToDotOperand(op, adaptor, rewriter);
    // }
    if (isaDistributedLayout(srcLayout) && isaDistributedLayout(dstLayout)) {
      return lowerDistributedToDistributed(op, adaptor, rewriter);
    }
    // if (isa<NvidiaMmaEncodingAttr>(srcLayout) &&
    //     isa<DotOperandEncodingAttr>(dstLayout)) {
    //   return lowerMmaToDotOperand(op, adaptor, rewriter);
    // }
    if (isa<SharedEncodingAttr>(srcLayout) && isaDistributedLayout(dstLayout)) {
      return lowerSharedToDistributed(op, adaptor, rewriter);
    }
    // TODO: to be implemented
    llvm_unreachable("unsupported layout conversion");
    return failure();
  }

private:
  SmallVector<Value>
  getMultiDimOffset(Attribute layout, Location loc,
                    ConversionPatternRewriter &rewriter, unsigned elemId,
                    RankedTensorType type,
                    ArrayRef<unsigned> multiDimCTAInRepId,
                    ArrayRef<unsigned> shapePerCTATile) const {
    auto shape = type.getShape();
    unsigned rank = shape.size();
    if (auto blockedLayout = dyn_cast<BlockedEncodingAttr>(layout)) {
      auto multiDimOffsetFirstElem =
          emitBaseIndexForLayout(loc, rewriter, blockedLayout, type, false);
      SmallVector<Value> multiDimOffset(rank);
      SmallVector<unsigned> multiDimElemId = getMultiDimIndex<unsigned>(
          elemId, getSizePerThread(layout), getOrder(layout));
      for (unsigned d = 0; d < rank; ++d) {
        multiDimOffset[d] =
            add(multiDimOffsetFirstElem[d],
                i32_val(multiDimCTAInRepId[d] * shapePerCTATile[d] +
                        multiDimElemId[d]));
      }
      return multiDimOffset;
    }
    if (auto sliceLayout = dyn_cast<SliceEncodingAttr>(layout)) {
      unsigned dim = sliceLayout.getDim();
      auto parentEncoding = sliceLayout.getParent();
      auto parentSizePerThread = getSizePerThread(parentEncoding);
      auto parentShape = sliceLayout.paddedShape(shape);
      auto parentTy = RankedTensorType::get(parentShape, type.getElementType(),
                                            parentEncoding);
      auto offsets = emitOffsetForLayout(layout, type);
      auto parentOffset = emitOffsetForLayout(parentEncoding, parentTy);
      SmallVector<int> idxs;
      for (SmallVector<unsigned> off : offsets) {
        off.insert(off.begin() + dim, 0);
        auto it = std::find(parentOffset.begin(), parentOffset.end(), off);
        idxs.push_back(std::distance(parentOffset.begin(), it));
      }
      auto multiDimOffsetParent = getMultiDimOffset(
          parentEncoding, loc, rewriter, idxs[elemId], parentTy,
          sliceLayout.paddedShape(multiDimCTAInRepId),
          sliceLayout.paddedShape(shapePerCTATile));
      SmallVector<Value> multiDimOffset(rank);
      for (unsigned d = 0; d < rank + 1; ++d) {
        if (d == dim)
          continue;
        unsigned slicedD = d < dim ? d : (d - 1);
        multiDimOffset[slicedD] = multiDimOffsetParent[d];
      }
      return multiDimOffset;
    }

    llvm_unreachable("unexpected layout in getMultiDimOffset");
  }

  SmallVector<Value>
  getWrappedMultiDimOffset(ConversionPatternRewriter &rewriter, Location loc,
                           ArrayRef<Value> multiDimOffset,
                           ArrayRef<unsigned> shape,
                           SmallVector<unsigned> shapePerCTATile,
                           SmallVector<int64_t> shapePerCTA) const {
    unsigned rank = shape.size();
    SmallVector<Value> multiDimOffsetWrapped(rank);
    for (unsigned d = 0; d < rank; ++d) {
      if (shapePerCTATile[d] > shapePerCTA[d])
        multiDimOffsetWrapped[d] = urem(multiDimOffset[d], i32_val(shape[d]));
      else
        multiDimOffsetWrapped[d] = multiDimOffset[d];
    }
    return multiDimOffsetWrapped;
  }

  // shared memory rd/st for blocked or mma layout with data padding
  void processReplica(Location loc, ConversionPatternRewriter &rewriter,
                      bool stNotRd, RankedTensorType type,
                      ArrayRef<unsigned> numCTAsEachRep,
                      ArrayRef<unsigned> multiDimRepId, unsigned vec,
                      ArrayRef<unsigned> paddedRepShape,
                      ArrayRef<unsigned> origRepShape,
                      ArrayRef<unsigned> outOrd, SmallVector<Value> &vals,
                      Value smemBase) const {
    auto accumNumCTAsEachRep = product<unsigned>(numCTAsEachRep);
    auto layout = type.getEncoding();
    auto rank = type.getRank();
    auto sizePerThread = getSizePerThread(layout);
    auto accumSizePerThread = product<unsigned>(sizePerThread);
    SmallVector<unsigned> numCTATiles(rank);
    auto shapePerCTATile = getShapePerCTATile(layout);
    auto shapePerCTA = getShapePerCTA(layout, type.getShape());
    auto order = getOrder(layout);
    for (unsigned d = 0; d < rank; ++d) {
      numCTATiles[d] = ceil<unsigned>(shapePerCTA[d], shapePerCTATile[d]);
    }
    auto elemTy = type.getElementType();
    bool isInt1 = elemTy.isInteger(1);
    bool isPtr = isa<triton::PointerType>(elemTy);
    auto llvmElemTyOrig = getTypeConverter()->convertType(elemTy);
    if (isInt1)
      elemTy = IntegerType::get(elemTy.getContext(), 8);
    else if (isPtr)
      elemTy = IntegerType::get(elemTy.getContext(), 64);

    auto llvmElemTy = getTypeConverter()->convertType(elemTy);

    for (unsigned ctaId = 0; ctaId < accumNumCTAsEachRep; ++ctaId) {
      auto multiDimCTAInRepId =
          getMultiDimIndex<unsigned>(ctaId, numCTAsEachRep, order);
      SmallVector<unsigned> multiDimCTAId(rank);
      for (const auto &it : llvm::enumerate(multiDimCTAInRepId)) {
        auto d = it.index();
        multiDimCTAId[d] = multiDimRepId[d] * numCTAsEachRep[d] + it.value();
      }

      auto linearCTAId =
          getLinearIndex<unsigned>(multiDimCTAId, numCTATiles, order);
      // TODO: This is actually redundant index calculation, we should
      //       consider of caching the index calculation result in case
      //       of performance issue observed.
      for (unsigned elemId = 0; elemId < accumSizePerThread; elemId += vec) {
        SmallVector<Value> multiDimOffset =
            getMultiDimOffset(layout, loc, rewriter, elemId, type,
                              multiDimCTAInRepId, shapePerCTATile);
        SmallVector<Value> multiDimOffsetWrapped = getWrappedMultiDimOffset(
            rewriter, loc, multiDimOffset, origRepShape, shapePerCTATile,
            shapePerCTA);
        Value offset = linearize(rewriter, loc, multiDimOffsetWrapped,
                                 paddedRepShape, outOrd);
        auto elemPtrTy = ptr_ty(llvmElemTy, spirv::StorageClass::Workgroup);
        Value ptr = gep(elemPtrTy, bitcast(smemBase, elemPtrTy), offset);
        if (vec == 1) {
          if (stNotRd) {
            auto currVal = vals[elemId + linearCTAId * accumSizePerThread];
            if (isInt1) {
              // spriv::UConvert doesn't support i1
              currVal = select(currVal, int_val(8, 1), int_val(8, 0));
            } else if (isPtr)
              currVal = ptrtoint(llvmElemTy, currVal);
            store(currVal, ptr);
          } else {
            Value currVal = load(ptr);
            if (isInt1)
              currVal = icmp_ne(currVal,
                                rewriter.create<spirv::ConstantOp>(
                                    loc, i8_ty, rewriter.getI8IntegerAttr(0)));
            else if (isPtr)
              currVal = inttoptr(llvmElemTyOrig, currVal);
            vals[elemId + linearCTAId * accumSizePerThread] = currVal;
          }
        } else {
          auto vecTy = struct_ty(SmallVector<Type>(vec, llvmElemTy));
          ptr = bitcast(ptr, ptr_ty(vecTy, spirv::StorageClass::Workgroup));
          if (stNotRd) {
            Value valVec = undef(vecTy);
            for (unsigned v = 0; v < vec; ++v) {
              auto currVal =
                  vals[elemId + linearCTAId * accumSizePerThread + v];
              if (isInt1) {
                // spriv::UConvert doesn't support i1
                currVal = select(currVal, int_val(8, 1), int_val(8, 0));
              } else if (isPtr)
                currVal = ptrtoint(llvmElemTy, currVal);

              valVec = insert_val(vecTy, currVal, valVec,
                                  rewriter.getI32ArrayAttr(v));
            }
            store(valVec, ptr);
          } else {
            Value valVec = load(ptr);
            for (unsigned v = 0; v < vec; ++v) {
              Value currVal =
                  extract_val(llvmElemTy, valVec, rewriter.getI32ArrayAttr(v));
              if (isInt1)
                currVal = icmp_ne(
                    currVal, rewriter.create<spirv::ConstantOp>(
                                 loc, i8_ty, rewriter.getI8IntegerAttr(0)));
              else if (isPtr)
                currVal = inttoptr(llvmElemTyOrig, currVal);
              vals[elemId + linearCTAId * accumSizePerThread + v] = currVal;
            }
          }
        }
      }
    }
  }

  // The MMAV1's result is quite different from the existing "Replica"
  // structure, add a new simple but clear implementation for it to avoid
  // modifying the logic of the existing one.
  void processReplicaForMMAV1(Location loc, ConversionPatternRewriter &rewriter,
                              bool stNotRd, RankedTensorType type,
                              ArrayRef<unsigned> multiDimRepId, unsigned vec,
                              ArrayRef<unsigned> paddedRepShape,
                              ArrayRef<unsigned> outOrd,
                              SmallVector<Value> &vals, Value smemBase,
                              ArrayRef<int64_t> shape,
                              bool isDestMma = false) const {
    assert(0 && "no mma support yet");
  }

  // blocked/mma -> blocked/mma.
  // Data padding in shared memory to avoid bank conflict.
  LogicalResult
  lowerDistributedToDistributed(triton::gpu::ConvertLayoutOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<RankedTensorType>(dst.getType());
    auto typeConverter = getTypeConverter();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    if (product(srcTy.getShape()) == 1) {
      auto inVals = typeConverter->unpackLLElements(loc, adaptor.getSrc(),
                                                    rewriter, srcTy);
      SmallVector<Value> outVals(getTotalElemsPerThread(dstTy), inVals[0]);
      Value result =
          typeConverter->packLLElements(loc, outVals, rewriter, dstTy);
      rewriter.replaceOp(op, result);
      return success();
    }

    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getOperation());
    auto elemPtrTy = ptr_ty(llvmElemTy, spirv::StorageClass::Workgroup);
    smemBase = bitcast(smemBase, elemPtrTy);
    auto shape = dstTy.getShape();
    unsigned rank = dstTy.getRank();
    SmallVector<unsigned> numReplicates(rank);
    SmallVector<unsigned> inNumCTAsEachRep(rank);
    SmallVector<unsigned> outNumCTAsEachRep(rank);
    SmallVector<unsigned> inNumCTAs(rank);
    SmallVector<unsigned> outNumCTAs(rank);
    auto srcShapePerCTATile = getShapePerCTATile(srcLayout, srcTy.getShape());
    auto dstShapePerCTATile = getShapePerCTATile(dstLayout, shape);
    auto shapePerCTA = getShapePerCTA(srcLayout, shape);

    // For Volta, all the coords for a CTA are calculated.
    bool isSrcMmaV1{}, isDstMmaV1{};
    if (auto mmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(srcLayout)) {
      isSrcMmaV1 = mmaLayout.isVolta();
    }
    if (auto sliceLayout = dyn_cast<SliceEncodingAttr>(srcLayout)) {
      isSrcMmaV1 =
          isa<NvidiaMmaEncodingAttr>(sliceLayout.getParent()) &&
          cast<NvidiaMmaEncodingAttr>(sliceLayout.getParent()).isVolta();
    }
    if (auto mmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(dstLayout)) {
      isDstMmaV1 = mmaLayout.isVolta();
    }
    if (auto sliceLayout = dyn_cast<SliceEncodingAttr>(dstLayout)) {
      isDstMmaV1 =
          isa<NvidiaMmaEncodingAttr>(sliceLayout.getParent()) &&
          cast<NvidiaMmaEncodingAttr>(sliceLayout.getParent()).isVolta();
    }

    for (unsigned d = 0; d < rank; ++d) {
      unsigned inPerCTA =
          std::min<unsigned>(shapePerCTA[d], srcShapePerCTATile[d]);
      unsigned outPerCTA =
          std::min<unsigned>(shapePerCTA[d], dstShapePerCTATile[d]);
      unsigned maxPerCTA = std::max(inPerCTA, outPerCTA);
      numReplicates[d] = ceil<unsigned>(shapePerCTA[d], maxPerCTA);
      inNumCTAsEachRep[d] = maxPerCTA / inPerCTA;
      outNumCTAsEachRep[d] = maxPerCTA / outPerCTA;
      assert(maxPerCTA % inPerCTA == 0 && maxPerCTA % outPerCTA == 0);
      inNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], inPerCTA);
      outNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], outPerCTA);
    }
    // Potentially we need to store for multiple CTAs in this replication
    auto accumNumReplicates = product<unsigned>(numReplicates);
    auto vals =
        typeConverter->unpackLLElements(loc, adaptor.getSrc(), rewriter, srcTy);
    unsigned inVec = 0;
    unsigned outVec = 0;
    auto origRepShape = getRepShapeForCvtLayout(op);
    auto paddedRepShape = getScratchConfigForCvtLayout(op, inVec, outVec);

    unsigned outElems = getTotalElemsPerThread(dstTy);
    auto outOrd = getOrder(dstLayout);
    SmallVector<Value> outVals(outElems);

    for (unsigned repId = 0; repId < accumNumReplicates; ++repId) {
      auto multiDimRepId =
          getMultiDimIndex<unsigned>(repId, numReplicates, outOrd);
      if (repId != 0)
        barrier();
      if (isa<BlockedEncodingAttr>(srcLayout) ||
          isa<SliceEncodingAttr>(srcLayout) ||
          isa<NvidiaMmaEncodingAttr>(srcLayout)) {
        if (isSrcMmaV1)
          processReplicaForMMAV1(loc, rewriter, /*stNotRd*/ true, srcTy,
                                 multiDimRepId, inVec, paddedRepShape, outOrd,
                                 vals, smemBase, shape);
        else

          processReplica(loc, rewriter, /*stNotRd*/ true, srcTy,
                         inNumCTAsEachRep, multiDimRepId, inVec, paddedRepShape,
                         origRepShape, outOrd, vals, smemBase);
      } else {
        assert(0 && "ConvertLayout with input layout not implemented");
        return failure();
      }

      barrier();
      if (isa<BlockedEncodingAttr>(dstLayout) ||
          isa<SliceEncodingAttr>(dstLayout) ||
          isa<NvidiaMmaEncodingAttr>(dstLayout)) {
        if (isDstMmaV1)
          processReplicaForMMAV1(loc, rewriter, /*stNotRd*/ false, dstTy,
                                 multiDimRepId, outVec, paddedRepShape, outOrd,
                                 outVals, smemBase, shape,
                                 /*isDestMma=*/true);
        else
          processReplica(loc, rewriter, /*stNotRd*/ false, dstTy,
                         outNumCTAsEachRep, multiDimRepId, outVec,
                         paddedRepShape, origRepShape, outOrd, outVals,
                         smemBase);
      } else {
        assert(0 && "ConvertLayout with output layout not implemented");
        return failure();
      }
    }

    Value result = typeConverter->packLLElements(loc, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

  LogicalResult
  lowerSharedToDistributed(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto srcShape = srcTy.getShape();
    auto dstTy = cast<RankedTensorType>(dst.getType());
    auto dstShape = dstTy.getShape();
    assert(dstShape.size() == 2 &&
           "Unexpected rank of ConvertLayout(shared->blocked)");
    auto srcSharedLayout = cast<SharedEncodingAttr>(srcTy.getEncoding());
    auto dstLayout = dstTy.getEncoding();
    auto inOrd = getOrder(srcSharedLayout);

    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(), rewriter);
    auto elemTy = getTypeConverter()->convertType(dstTy.getElementType());

    auto srcStrides =
        getStridesFromShapeAndOrder(srcShape, inOrd, loc, rewriter);
    auto dstIndices = emitIndices(loc, rewriter, dstLayout, dstTy);

    SmallVector<Value> outVals = loadSharedToDistributed(
        dst, dstIndices, src, smemObj, elemTy, loc, rewriter);

    Value result =
        getTypeConverter()->packLLElements(loc, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

  // blocked -> shared.
  // Swizzling in shared memory to avoid bank conflict. Normally used for
  // A/B operands of dots.
  LogicalResult
  lowerDistributedToShared(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto srcShape = srcTy.getShape();
    auto dstTy = cast<RankedTensorType>(dst.getType());
    auto dstShapePerCTA = triton::gpu::getShapePerCTA(dstTy);
    assert(srcShape.size() == 2 &&
           "Unexpected rank of ConvertLayout(blocked->shared)");
    auto srcLayout = srcTy.getEncoding();
    auto dstSharedLayout = cast<SharedEncodingAttr>(dstTy.getEncoding());
    auto inOrd = getOrder(srcLayout);
    auto outOrd = dstSharedLayout.getOrder();
    Value smemBase = getSharedMemoryBase(loc, rewriter, dst);
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto elemPtrTy = ptr_ty(getTypeConverter()->convertType(elemTy),
                            spirv::StorageClass::Workgroup);
    smemBase = bitcast(smemBase, elemPtrTy);

    auto dstStrides =
        getStridesFromShapeAndOrder(dstShapePerCTA, outOrd, loc, rewriter);
    auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcTy, false);
    storeDistributedToShared(src, adaptor.getSrc(), dstStrides, srcIndices, dst,
                             smemBase, elemTy, loc, rewriter);
    auto smemObj =
        SharedMemoryObject(smemBase, dstShapePerCTA, outOrd, loc, rewriter);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

  //   // mma -> dot_operand
  //   LogicalResult
  //   lowerMmaToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
  //                        ConversionPatternRewriter &rewriter) const {
  //     assert(0 && "no mma support yet");
  //   }
}; // namespace triton::gpu::ConvertLayoutOp

void populateConvertLayoutOpToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpSPIRVConversion>(
      typeConverter, context, allocation, indexCacheInfo, benefit);
  patterns.add<LocalLoadOpConversion>(typeConverter, context, allocation,
                                      indexCacheInfo, benefit);
}
