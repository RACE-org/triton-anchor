#include "HistogramOpToSPIRV.h"
using namespace mlir;
using namespace mlir::triton;

using ::mlir::spirv::delinearize;
using ::mlir::spirv::linearize;
using ::mlir::spirv::loadShared;
using ::mlir::spirv::shflSync;
using ::mlir::spirv::storeShared;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getTotalElemsPerThread;

static int log2Int(int64_t num) { return (num > 1) ? 1 + log2Int(num / 2) : 0; }

// Compute a histogram within a warp. This uses an algorithm by @apgoucher
// that does the following:
// Create a ballot for each bit of the bin index (there
// are only log2(num_bins) of these) and then apply bitwise operations to get
// the indicator functions for the bins owned by this particular thread, and
// only popcount those.
static SmallVector<Value>
computeWarpLevelHistogram(Location loc, RankedTensorType srcType,
                          SmallVector<Value> &srcValues, int numBins,
                          int numThreadPerWarp, Value threadId,
                          ConversionPatternRewriter &rewriter) {
  int numThreadPerWarpSub = 32; // due to ballot returns vector<4, int32>
  assert(numBins % numThreadPerWarp == 0 &&
         "numBins must be divisible by numThreadPerWarp");
  Value zero = i32_val(0);
  int numBits = log2Int(numBins);
  int numBitsLaneId = log2Int(numThreadPerWarp);
  unsigned numElementsPerThreads = triton::gpu::getTotalElemsPerThread(srcType);
  unsigned numThreadWithUniqueData =
      triton::gpu::getThreadsPerWarpWithUniqueData(srcType.getEncoding(),
                                                   srcType.getShape())[0];
  // The histogram is distributed across threads, each thread owns `numBins /
  // numThreadPerWarp` bins.
  SmallVector<Value> warpLevelHistogram(numBins / numThreadPerWarp, zero);
  for (int i = 0; i < numElementsPerThreads; ++i) {

    Value value = srcValues[i];
    SmallVector<Value> ballot128Bits;
    for (int j = 0; j < numBits; ++j) {
      Value bitSet = and_(value, i32_val(1 << j));
      Value cmp = icmp_ne(bitSet, zero);
      Value bit = rewriter.create<spirv::GroupNonUniformBallotOp>(
          loc, vec_ty(int_ty(32), 4), spirv::Scope::Subgroup,
          cmp); // result must be vector<4xi32>
      ballot128Bits.push_back(bit);
    }
    uint64_t fullMaskValue = 0xFFFFFFFF;
    Value fullMask = int_val(numThreadPerWarpSub, fullMaskValue);
    SmallVector<Value> ballot32Bits;

    for (int idx = 0; idx < 4; idx++) {
      // break as early as possible in order to avoid invalid calc
      if ((numThreadWithUniqueData <= 32 && idx > 0) ||
          (numThreadWithUniqueData <= 64 && idx > 1) ||
          (numThreadWithUniqueData <= 96 && idx > 2))
        break;

      ballot32Bits.clear();
      for (auto ballot128Bit : ballot128Bits) {
        ballot32Bits.push_back(extract_val(int_ty(32), ballot128Bit,
                                           rewriter.getI32ArrayAttr(idx)));
      }

      Value mask = fullMask;
      // If not all threads have unique data, mask out the redundant ones.
      if (numThreadWithUniqueData < ((idx + 1) * 32)) {
        mask = int_val(numThreadPerWarpSub,
                       (1ULL << (numThreadWithUniqueData - idx * 32)) - 1);
      }
      for (int i = 0; i < numBitsLaneId; i++) {
        Value updateMask =
            select(icmp_ne(and_(threadId, i32_val(1 << i)), zero),
                   int_val(numThreadPerWarpSub, 0), fullMask);
        mask = and_(
            mask, xor_(ballot32Bits[i + numBits - numBitsLaneId], updateMask));
      }
      // at this point, 'mask' tells you which elements are in a bin owned by
      // this thread.
      for (int k = 0; k < warpLevelHistogram.size(); k++) {
        Value binMask = mask;
        for (int j = 0; j < numBits - numBitsLaneId; j++) {
          Value updateMask = int_val(numThreadPerWarpSub,
                                     ((k & (1 << j)) ? 0 : fullMaskValue));
          binMask = and_(binMask, xor_(ballot32Bits[j], updateMask));
        }
        // at this point, 'bin_mask' tells you which elements are in the kth
        // bin owned by this thread.
        Value bitCount = rewriter.create<spirv::BitCountOp>(
            loc, int_ty(numThreadPerWarpSub), binMask);

        warpLevelHistogram[k] = add(warpLevelHistogram[k], bitCount);
      }
    }
  }
  return warpLevelHistogram;
}

static void atomicAdd(Value ptr, Value val, Location loc,
                      ConversionPatternRewriter &rewriter) {
  rewriter.create<spirv::AtomicIAddOp>(
      loc, val.getType(), ptr,
      spirv::Scope::Workgroup, // it's a atomic op on Shared memory level
      spirv::MemorySemantics::AcquireRelease, val);
}

static SmallVector<Value> computeCrossWarpHistogram(
    Location loc, ConversionPatternRewriter &rewriter, RankedTensorType srcType,
    Value baseSharedMemPtr, const SmallVector<Value> &warpLevelHistogram,
    int numBins, int numThreadPerWarp, const SmallVector<Value> &indices,
    Value threadId, int numWarps) {
  SmallVector<Value> histogramValues;
  unsigned numWarpsWithUniqueData =
      mlir::triton::gpu::getWarpsPerCTAWithUniqueData(srcType.getEncoding(),
                                                      srcType.getShape())[0];
  Value laneId = and_(threadId, i32_val(numThreadPerWarp - 1));
  // Initialize the shared memory with zeros.
  int64_t numElementPerThread =
      ceil<int64_t>(numBins, numThreadPerWarp * numWarps);
  for (int i = 0; i < numElementPerThread; ++i) {
    Value offset = add(threadId, i32_val((i * numWarps * numThreadPerWarp)));
    offset = urem(offset, i32_val(numBins));
    Value sharedMemPtr =
        gep(baseSharedMemPtr.getType(), baseSharedMemPtr, offset);
    store(i32_val(0), sharedMemPtr);
  }
  barrier();
  Block *afterAtomics = nullptr;
  // If some warps have replicated data we need to skip those warps when
  // accumulating.
  if (numWarpsWithUniqueData < numWarps) {
    Block *currentBlock = rewriter.getInsertionBlock();
    afterAtomics =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *atomicBlock = rewriter.createBlock(afterAtomics);
    rewriter.setInsertionPointToEnd(currentBlock);
    Value cond =
        icmp_ult(threadId, i32_val(numWarpsWithUniqueData * numThreadPerWarp));
    rewriter.create<mlir::cf::CondBranchOp>(loc, cond, atomicBlock,
                                            afterAtomics);
    rewriter.setInsertionPointToStart(atomicBlock);
  }
  // Apply atomic add to update the histogram in shared memory.
  for (int i = 0; i < warpLevelHistogram.size(); ++i) {
    Value warpLevelHistogramValue = warpLevelHistogram[i];
    Value offset =
        add(mul(laneId, i32_val(warpLevelHistogram.size())), i32_val(i));
    Value sharedMemPtr =
        gep(baseSharedMemPtr.getType(), baseSharedMemPtr, offset);
    atomicAdd(sharedMemPtr, warpLevelHistogramValue, loc, rewriter);
  }
  if (afterAtomics) {
    rewriter.create<mlir::cf::BranchOp>(loc, afterAtomics);
    rewriter.setInsertionPointToStart(afterAtomics);
  }
  barrier();
  // load the histogram to register with the right layout.
  for (Value index : indices) {
    Value sharedMemPtr =
        gep(baseSharedMemPtr.getType(), baseSharedMemPtr, index);
    Value val = load(i32_ty, sharedMemPtr);
    histogramValues.push_back(val);
  }
  return histogramValues;
}

struct HistogramOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::HistogramOp> {
public:
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::HistogramOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::HistogramOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getSrc();
    auto typeConverter = getTypeConverter();
    SmallVector<Value> srcValues =
        typeConverter->unpackLLElements(loc, input, rewriter, input.getType());
    int numBins = op.getType().getDimSize(0);
    auto mod = op->getParentOfType<ModuleOp>();
    int numThreadsPerWarp =
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    assert(numThreadsPerWarp == 128 && "FantGPU supports 128 threads per warp");

    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    // Pad out the bins so that we have at least one bin per thread within a
    // warp.
    numBins = std::max(numBins, numThreadsPerWarp);
    Value threadId = getThreadId(rewriter, loc);
    auto srcType = op.getSrc().getType();
    // First compute a warp local histogram based on values owned by each warps.
    SmallVector<Value> warpLevelHistogram =
        computeWarpLevelHistogram(loc, srcType, srcValues, numBins,
                                  numThreadsPerWarp, threadId, rewriter);

    // Then use atomic to update the histogram in shared memory.
    // TODO: we could skip this for cases with num_warps=1 as long as we can
    // generate the right layout. Currently the warp level histogram generates
    // data in the default blocked layout.
    auto TensorTy = mlir::dyn_cast<RankedTensorType>(op.getResult().getType());
    Type valueElemTy =
        TensorTy ? typeConverter->convertType(TensorTy.getElementType())
                 : op.getResult().getType();
    Value baseSharedMemPtr =
        getSharedMemoryBase(loc, rewriter, op.getOperation());
    baseSharedMemPtr = bitcast(
        baseSharedMemPtr, ptr_ty(valueElemTy, spirv::StorageClass::Workgroup));
    auto dstType = op.getType();
    Attribute dstEncoding = dstType.getEncoding();
    auto indices =
        emitIndices(op.getLoc(), rewriter, dstEncoding, dstType, true);
    SmallVector<Value> innerDimIndices;
    for (int i = 0; i < indices.size(); ++i)
      innerDimIndices.push_back(indices[i][0]);
    SmallVector<Value> histogramValue = computeCrossWarpHistogram(
        loc, rewriter, srcType, baseSharedMemPtr, warpLevelHistogram, numBins,
        numThreadsPerWarp, innerDimIndices, threadId, numWarps);

    Value results = typeConverter->packLLElements(loc, histogramValue, rewriter,
                                                  op.getType());
    rewriter.replaceOp(op, results);
    return success();
  }

private:
};

void populateHistogramOpToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<HistogramOpSPIRVConversion>(typeConverter, context, allocation,
                                           indexCacheInfo, benefit);
}
