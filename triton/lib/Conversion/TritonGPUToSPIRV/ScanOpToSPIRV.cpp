#include "ScanOpToSPIRV.h"
#include "TritonGPUToSPIRVBase.h"
#include "triton/Analysis/Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::spirv::delinearize;
using ::mlir::spirv::linearize;
using ::mlir::spirv::shflIdxSync;
using ::mlir::spirv::shflUpSync;
using ::mlir::spirv::storeShared;
using ::mlir::triton::gpu::getTotalElemsPerThread;
// Apply the region of the scan op to the acc and cur values and update acc
// inplace with the result.
static SmallVector<Value> accumulate(ConversionPatternRewriter &rewriter, Region &combineOp,
                        ValueRange acc, ValueRange cur) {

  if (acc.size() == 0) return cur;
  assert(acc.size() == cur.size());
  // if (!acc) {
  //   return cur;
  // }
  // if (!cur) {
  //   return acc;
  // }

  // Create a new copy of the reduce block, and inline it
  Block *currentBlock = rewriter.getBlock();
  Region &parent = *currentBlock->getParent();
  rewriter.cloneRegionBefore(combineOp, &parent.front());
  auto &newScan = parent.front();
  auto returnOp = dyn_cast<triton::ScanReturnOp>(newScan.getTerminator());
  // llvm::SmallVector<Value> combineArgs = {acc, cur};
  SmallVector<Value> combineArgs(2 * acc.size());
  for (unsigned i = 0; i < acc.size(); ++i) {
    combineArgs[i] = acc[i];
    combineArgs[acc.size() + i] = cur[i];
  }

  rewriter.inlineBlockBefore(&newScan, &*rewriter.getInsertionPoint(),
                             combineArgs);
  auto results = returnOp.getResult();
  // Value ret = results[0];
  // Delete the terminator, which is no longer used
  rewriter.eraseOp(returnOp);
  return results;
}

// Scan a contiguous elements within a thread and update `srcValues` in place.
static void scanThreadContiguousElements(SmallVector<SmallVector<Value>> &srcValues,
                                         ConversionPatternRewriter &rewriter,
                                         ScanLoweringHelper &helper) {
  // Depending on layout contiguous elements along axis dim may not be
  // contiguous in srcValues. Keep track of what elements belong to the same
  // chunk of contiguous elements.
  unsigned scanElementsPerThreads = helper.getAxisNumElementsPerThread();
  unsigned numChunks = srcValues.size() / scanElementsPerThreads;
  unsigned stride = helper.getAxisElementStride();
  SmallVector<SmallVector<Value>> accs(numChunks);
  for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
    unsigned accIndex = (srcIndex % stride) +
                        ((srcIndex / stride) / scanElementsPerThreads) * stride;

    accs[accIndex] = accumulate(rewriter, helper.getCombineOp(), accs[accIndex],
                                srcValues[srcIndex]);
    srcValues[srcIndex] = accs[accIndex];
  }
}

// Apply a scan across threads of the warp for the last element of each
// contiguous group of elements.
static void warpScan(SmallVector<SmallVector<Value>> &srcValues,
                     ConversionPatternRewriter &rewriter,
                     ScanLoweringHelper &helper, Value laneIdAxis) {
  Location loc = helper.getLoc();
  unsigned scanElementsPerThreads = helper.getAxisNumElementsPerThread();
  unsigned elementStride = helper.getAxisElementStride();
  unsigned threadStride = helper.getAxisThreadStride();
  unsigned scanDim = helper.getAxisNumThreadsPerWarpWithUniqueData();
  for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
    unsigned elementIdx = (srcIndex / elementStride) % scanElementsPerThreads;
    // Only consider the last element of each contiguous chunk of elements.
    if (elementIdx != scanElementsPerThreads - 1)
      continue;
    // Reduce within warps.
    SmallVector<Value> acc = srcValues[srcIndex];
    for (unsigned i = 1; i <= (scanDim) / 2; i = i << 1) {
      SmallVector<Value> shfl(acc.size());
      for (unsigned j = 0; j < acc.size(); ++j) {
        shfl[j] = shflUpSync(loc, rewriter, acc[j], i * threadStride);
      }
      SmallVector<Value> tempAcc = accumulate(rewriter, helper.getCombineOp(), shfl, acc);
      // Value shfl = shflUpSync(loc, rewriter, acc, i * threadStride);
      // Value tempAcc = accumulate(rewriter, helper.getCombineOp(), shfl, acc);
      Value mask = icmp_slt(laneIdAxis, i32_val(i));
      // acc = select(mask, acc, tempAcc);
      for (unsigned j = 0; j < acc.size(); ++j) {
        acc[j] = select(mask, acc[j], tempAcc[j]);
      }
    }
    srcValues[srcIndex] = acc;
  }
}

// For each set of contiguous elements within a thread we store the partial
// reduction into shared memory. Each parallel scan and each warp will store its
// own partial reductions. The shared memory is organized as follow:
//          -----------------------------------------------------------------
// chunk 0: | acc[0] warp 0 | acc[1] warp 0 | acc[0] warp 1 | acc[1] warp 1 |
// chunk 1: | acc[0] warp 0 | acc[1] warp 0 | acc[0] warp 1 | acc[1] warp 1 |
static void storeWarpAccumulator(SmallVector<SmallVector<Value>> &srcValues,
                                 ConversionPatternRewriter &rewriter,
                                 ScanLoweringHelper &helper, Value laneId,
                                 Value warpId, SmallVector<Value> smemBases,
                                 SmallVector<Type> smemTypes,
                                 Value parallelLaneId) {
  Location loc = helper.getLoc();
  unsigned scanElementsPerThreads = helper.getAxisNumElementsPerThread();
  unsigned scanDim = helper.getAxisNumThreadsPerWarpWithUniqueData();
  unsigned numParallelLane = helper.getNonAxisNumThreadsPerCTA();
  unsigned axisNumWarps = helper.getAxisNumWarpsWithUniqueData();
  unsigned chunkId = 0;
  unsigned elementStride = helper.getAxisElementStride();
  for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
    unsigned elementIdx = (srcIndex / elementStride) % scanElementsPerThreads;
    // Only consider the last element of each contiguous chunk of elements.
    if (elementIdx != scanElementsPerThreads - 1)
      continue;
    auto lastElement = srcValues[srcIndex];
    Value mask = icmp_eq(laneId, i32_val(scanDim - 1));
    Value index = add(parallelLaneId, mul(warpId, i32_val(numParallelLane)));
    index = add(index, i32_val(chunkId * numParallelLane * axisNumWarps));
    // Value writePtr = gep(baseSharedMemPtr.getType(), baseSharedMemPtr, index);
    // storeShared(rewriter, loc, writePtr, lastElement, mask);

    for (unsigned i = 0; i < lastElement.size(); ++i) {
      Value writePtr = gep(smemBases[i].getType(), smemBases[i], index);
      storeShared(rewriter, loc, writePtr, lastElement[i], mask);
    }
    chunkId++;
  }
}

// Read the partial reductions from shared memory from each chunk of contiguous
// elements for each warp and parallel scan. Then combine the partial reduction
// with the right elements. Within a given contiguous element chunk we update
// all the elements by accumulating the value from the last element of the
// reduced value from the previous lane.
static void AddPartialReduce(SmallVector<SmallVector<Value>> &srcValues,
                             ConversionPatternRewriter &rewriter,
                             ScanLoweringHelper &helper, SmallVector<Value> smemBases,
                             SmallVector<Type> smemTypes, Value warpId,
                             Value laneIdAxis, Value parallelLaneId) {
  Location loc = helper.getLoc();
  unsigned numParallelLane = helper.getNonAxisNumThreadsPerCTA();
  unsigned scanElementsPerThreads = helper.getAxisNumElementsPerThread();
  unsigned parallelElementsPerThread = helper.getNonAxisNumElementsPerThread();
  unsigned elementStride = helper.getAxisElementStride();
  unsigned threadStride = helper.getAxisThreadStride();
  unsigned axisNumWarps = helper.getAxisNumWarpsWithUniqueData();
  Value maskFirstWarp = icmp_eq(warpId, i32_val(0));
  Value maskFirstLane = icmp_eq(laneIdAxis, i32_val(0));
  Value maskFirstThread = and_(maskFirstWarp, maskFirstLane);
  struct Accumulator {
    SmallVector<Value> acc;
    SmallVector<Value> maskedAcc;
  };
  unsigned numScanBlocks = helper.getAxisNumBlocks();
  unsigned numParallelBlocks = helper.getNonAxisNumBlocks();
  assert(numScanBlocks * numParallelBlocks * parallelElementsPerThread *
             scanElementsPerThreads ==
         srcValues.size());
  SmallVector<Accumulator> accumulators(numParallelBlocks *
                                        parallelElementsPerThread);
  unsigned chunkId = 0;
  unsigned blockStride = helper.getAxisBlockStride();
  for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
    unsigned elementIdx = (srcIndex / elementStride) % scanElementsPerThreads;
    // Only consider the last element of each contiguous chunk of elements.
    if (elementIdx != scanElementsPerThreads - 1)
      continue;
    // Accumulate the partial reduction from shared memory. Decide which
    // accumulator to combine based on whether the elements belong to the same
    // dimension along axis.
    unsigned blockId = chunkId / parallelElementsPerThread;
    unsigned parallelBlockId =
        blockId % blockStride +
        ((blockId / blockStride) / numScanBlocks) * blockStride;
    unsigned accumulatorIndex = chunkId % parallelElementsPerThread +
                                parallelBlockId * parallelElementsPerThread;
    Accumulator &accumulator = accumulators[accumulatorIndex];
    unsigned axisBlockId = (blockId / blockStride) % numScanBlocks;
    for (unsigned i = 0; i < axisNumWarps; ++i) {
      Value index = add(parallelLaneId, i32_val(numParallelLane *
                                                (i + chunkId * axisNumWarps)));
      // Value ptr = gep(sharedMemoryPtr.getType(), sharedMemoryPtr, index);
      // Value partialReduce = load(ptr);
      // if (!accumulator.acc) {
      SmallVector<Value> partialReduce(helper.getNumOperands());
      for (unsigned j = 0; j < helper.getNumOperands(); ++j) {
        auto elemTy = smemTypes[j];
        Value ptr = gep(smemBases[j].getType(), smemBases[j], index);
        partialReduce[j] = load(elemTy, ptr);
      }

      if (accumulator.acc.size() == 0) {
        accumulator.acc = partialReduce;
        accumulator.maskedAcc = partialReduce;
        continue;
      }
      accumulator.acc = accumulate(rewriter, helper.getCombineOp(),
                                   accumulator.acc, partialReduce);
      Value mask = icmp_slt(warpId, i32_val(i + 1));
      // accumulator.maskedAcc =
      //     select(mask, accumulator.maskedAcc, accumulator.acc);
      for (unsigned j = 0; j < helper.getNumOperands(); ++j) {
        accumulator.maskedAcc[j] = select(mask, accumulator.maskedAcc[j], accumulator.acc[j]);
      }
    }
    auto temp = accumulate(rewriter, helper.getCombineOp(),
                            accumulator.maskedAcc, srcValues[srcIndex]);
    if (axisBlockId == 0) {
      // For the first warp and first chunk we don't have anything to
      // accumulate.
      // temp = select(maskFirstWarp, srcValues[srcIndex], temp);
      auto val = srcValues[srcIndex];
      for (unsigned i = 0; i < helper.getNumOperands(); ++i) {
        temp[i] = select(maskFirstWarp, val[i], temp[i]);
      }
    }
    srcValues[srcIndex] = temp;
    // Update the rest of the contiguous elements.
    // Value lastElement =
    //     shflUpSync(loc, rewriter, srcValues[srcIndex], threadStride);
    // lastElement = select(maskFirstLane, accumulator.maskedAcc, lastElement);
    SmallVector<Value> lastElement(helper.getNumOperands());
    for (unsigned i = 0; i < helper.getNumOperands(); ++i) {
      auto elem = shflUpSync(loc, rewriter, temp[i], threadStride);
      lastElement[i] = select(maskFirstLane, accumulator.maskedAcc[i], elem);
    }
    for (unsigned i = 1; i < scanElementsPerThreads; ++i) {
      auto laneValue = srcValues[srcIndex - i * elementStride];
      laneValue =
          accumulate(rewriter, helper.getCombineOp(), lastElement, laneValue);
      if (axisBlockId == 0) {
        // For the first warp and first chunk we don't have anything to
        // accumulate.
        for (unsigned j = 0; j < helper.getNumOperands(); ++j) {
          laneValue[j] = select(maskFirstThread,
                             srcValues[srcIndex - i * elementStride][j], laneValue[j]);
        }
        // laneValue = select(maskFirstThread,
        //                    srcValues[srcIndex - i * elementStride], laneValue);
      }
      srcValues[srcIndex - i * elementStride] = laneValue;
    }
    // For the next chunk start back from the value containing the
    // accumulated value of all the warps.
    accumulator.maskedAcc = accumulator.acc;
    chunkId++;
  }
}

static void AddPartialReduceOneWarp(SmallVector<SmallVector<Value>> &srcValues,
                                    ConversionPatternRewriter &rewriter,
                                    ScanLoweringHelper &helper, Value warpId,
                                    Value laneIdAxis, Value laneIdLast) {
  Location loc = helper.getLoc();
  unsigned scanElementsPerThreads = helper.getAxisNumElementsPerThread();
  unsigned parallelElementsPerThread = helper.getNonAxisNumElementsPerThread();
  unsigned elementStride = helper.getAxisElementStride();
  unsigned threadStride = helper.getAxisThreadStride();
  unsigned axisNumWarps = helper.getAxisNumWarpsWithUniqueData();
  unsigned numParallelLane = helper.getNonAxisNumThreadsPerCTA();
  unsigned scanDim = helper.getAxisNumThreadsPerWarpWithUniqueData();
  Value maskFirstWarp = icmp_eq(warpId, i32_val(0));
  Value maskFirstLane = icmp_eq(laneIdAxis, i32_val(0));
  Value maskFirstThread = and_(maskFirstWarp, maskFirstLane);
  unsigned numScanBlocks = helper.getAxisNumBlocks();
  unsigned numParallelBlocks = helper.getNonAxisNumBlocks();
  assert(numScanBlocks * numParallelBlocks * parallelElementsPerThread *
             scanElementsPerThreads ==
         srcValues.size());
  SmallVector<SmallVector<Value>> accumulators(numParallelBlocks *
                                  parallelElementsPerThread);
  unsigned chunkId = 0;
  unsigned blockStride = helper.getAxisBlockStride();
  for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
    unsigned elementIdx = (srcIndex / elementStride) % scanElementsPerThreads;
    // Only consider the last element of each contiguous chunk of elements.
    if (elementIdx != scanElementsPerThreads - 1)
      continue;
    unsigned blockId = chunkId / parallelElementsPerThread;
    unsigned parallelBlockId =
        blockId % blockStride +
        ((blockId / blockStride) / numScanBlocks) * blockStride;
    unsigned accumulatorIndex = chunkId % parallelElementsPerThread +
                                parallelBlockId * parallelElementsPerThread;
    auto &accumulator = accumulators[accumulatorIndex];
    unsigned axisBlockId = (blockId / blockStride) % numScanBlocks;
    if (axisBlockId == 0) // First chunk and first block
      accumulator = srcValues[srcIndex];
    else
      srcValues[srcIndex] = accumulate(rewriter, helper.getCombineOp(),
                                       accumulator, srcValues[srcIndex]);
    // Update the rest of the contiguous elements.
    auto lastElement = srcValues[srcIndex];
    if (scanDim > 1) {
      // lastElement =
      //     shflUpSync(loc, rewriter, srcValues[srcIndex], threadStride);
      // lastElement = select(maskFirstLane, accumulator, lastElement);
      // if (numScanBlocks > 1)
      //   // Update accumulator with the value from the last lane.
      //   accumulator =
      //       shflIdxSync(loc, rewriter, srcValues[srcIndex], laneIdLast);
      for (unsigned i = 0; i < helper.getNumOperands(); ++i) {
        lastElement[i] =
          shflUpSync(loc, rewriter, srcValues[srcIndex][i], threadStride);
        lastElement[i] = select(maskFirstLane, accumulator[i], lastElement[i]);
        if (numScanBlocks > 1)
          // Update accumulator with the value from the last lane.
          accumulator[i] =
              shflIdxSync(loc, rewriter, srcValues[srcIndex][i], laneIdLast);
      }
    }
    for (unsigned i = 1; i < scanElementsPerThreads; ++i) {
      auto laneValue = srcValues[srcIndex - i * elementStride];
      laneValue =
          accumulate(rewriter, helper.getCombineOp(), lastElement, laneValue);
      if (axisBlockId == 0) {
        for (unsigned j = 0; j < helper.getNumOperands(); ++j) {
          // For the first warp and first chunk we don't have anything to
          // accumulate.
          laneValue[j] = select(maskFirstThread,
                             srcValues[srcIndex - i * elementStride][j], laneValue[j]);
        }
      }
        // // For the first warp and first chunk we don't have anything to
        // // accumulate.
        // laneValue = select(maskFirstThread,
        //                    srcValues[srcIndex - i * elementStride], laneValue);
      srcValues[srcIndex - i * elementStride] = laneValue;
    }
    // For the next chunk start back from the value containing the
    // accumulated value of all the warps.
    chunkId++;
  }
}

namespace {
struct ScanOpConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::ScanOp> {
public:
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::ScanOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (succeeded(emitFastScan(op, adaptor, rewriter)))
      return success();
    return failure();
  }

private:
  SmallVector<Value> getMultiDimLaneId(ConversionPatternRewriter &rewriter,
                                       ScanLoweringHelper &helper,
                                       Value laneId) const;
  SmallVector<Value> getMultiDimWarpId(ConversionPatternRewriter &rewriter,
                                       ScanLoweringHelper &helper,
                                       Value warpId) const;
  std::tuple<Value, Value, Value>
  getDelinearizedIds(ConversionPatternRewriter &rewriter,
                     ScanLoweringHelper &helper, Value laneId,
                     Value warpId) const;
  Type getElementPtrType(triton::ScanOp op, int i) const {
    auto ty = op.getInputTypes()[i].getElementType();
    auto spirvElemTy = getTypeConverter()->convertType(ty);
    return spirv::PointerType::get(spirvElemTy, spirv::StorageClass::Workgroup);
  }

  SmallVector<Value> getSmemBases(ScanLoweringHelper &helper, triton::ScanOp op,
                                  unsigned elems,
                                  ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    // unsigned elems = product<unsigned>(smemShape);
    // indices will store the index of the op operands in descending order
    // of their bitwidths
    std::vector<unsigned> indices(op.getNumOperands());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](unsigned i, unsigned j) {
      return op.getElementTypes()[i].getIntOrFloatBitWidth() >
             op.getElementTypes()[j].getIntOrFloatBitWidth();
    });
    // Assign base index to each operand in their order in indices
    std::map<unsigned, Value> indexToBase;
    indexToBase[indices[0]] =
        bitcast(getSharedMemoryBase(loc, rewriter, op.getOperation()),
                getElementPtrType(op, indices[0]));
    for (unsigned i = 1; i < op.getNumOperands(); ++i) {
      indexToBase[indices[i]] =
          bitcast(gep(getElementPtrType(op, indices[i - 1]),
                      indexToBase[indices[i - 1]], i32_val(elems)),
                  getElementPtrType(op, indices[i]));
    }
    // smemBases[k] is the base pointer for the k-th operand
    SmallVector<Value> smemBases(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      smemBases[i] = indexToBase[i];
    }
    return smemBases;
  }

  Type getElementType(triton::ScanOp op, int i) const {
    auto ty = op.getInputTypes()[i].getElementType();
    return getTypeConverter()->convertType(ty);
  }
  LogicalResult emitFastScan(triton::ScanOp op, triton::ScanOpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const;
};

SmallVector<Value>
ScanOpConversion::getMultiDimLaneId(ConversionPatternRewriter &rewriter,
                                    ScanLoweringHelper &helper,
                                    Value laneId) const {
  auto loc = helper.getLoc();
  unsigned axis = helper.getAxis();
  auto srcEncoding = helper.getEncoding();

  auto threadsPerWarp = triton::gpu::getThreadsPerWarp(srcEncoding);
  auto warpsPerCTA = triton::gpu::getWarpsPerCTA(srcEncoding);
  auto order = triton::gpu::getOrder(srcEncoding);
  return delinearize(rewriter, loc, laneId, threadsPerWarp, order);
}

SmallVector<Value>
ScanOpConversion::getMultiDimWarpId(ConversionPatternRewriter &rewriter,
                                    ScanLoweringHelper &helper,
                                    Value warpId) const {
  auto loc = helper.getLoc();
  unsigned axis = helper.getAxis();
  auto srcEncoding = helper.getEncoding();

  auto threadsPerWarp = triton::gpu::getThreadsPerWarp(srcEncoding);
  auto warpsPerCTA = triton::gpu::getWarpsPerCTA(srcEncoding);
  auto order = triton::gpu::getOrder(srcEncoding);
  return delinearize(rewriter, loc, warpId, warpsPerCTA, order);
}

// Break up the threadId into lane and warp id along the scan dimension and
// compute a flat id for the parallel dimensions.
std::tuple<Value, Value, Value>
ScanOpConversion::getDelinearizedIds(ConversionPatternRewriter &rewriter,
                                     ScanLoweringHelper &helper, Value laneId,
                                     Value warpId) const {
  auto loc = helper.getLoc();
  unsigned axis = helper.getAxis();
  auto srcEncoding = helper.getEncoding();

  auto threadsPerWarp = triton::gpu::getThreadsPerWarp(srcEncoding);
  auto warpsPerCTA = triton::gpu::getWarpsPerCTA(srcEncoding);
  auto order = triton::gpu::getOrder(srcEncoding);
  SmallVector<Value> multiDimLaneId =
      delinearize(rewriter, loc, laneId, threadsPerWarp, order);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, warpsPerCTA, order);

  Value laneIdAxis = multiDimLaneId[axis];
  Value warpIdAxis = multiDimWarpId[axis];

  multiDimLaneId[axis] = i32_val(0);
  threadsPerWarp[axis] = 1;
  Value laneIdParallel =
      linearize(rewriter, loc, multiDimLaneId, threadsPerWarp, order);
  multiDimWarpId[axis] = i32_val(0);
  warpsPerCTA[axis] = 1;
  Value warpIdParallel =
      linearize(rewriter, loc, multiDimWarpId, warpsPerCTA, order);
  Value flatIdParallel =
      add(laneIdParallel,
          mul(warpIdParallel, i32_val(helper.getNonAxisNumThreadsPerWarp())));
  return std::make_tuple(laneIdAxis, warpIdAxis, flatIdParallel);
}

SmallVector<SmallVector<Value>>
unpackInputs(Location loc, triton::ScanOp op, triton::ScanOpAdaptor adaptor,
              ConversionPatternRewriter &rewriter,
              TritonGPUToSPIRVTypeConverter &converter) {
  auto types = op.getInputTypes();
  auto operands = adaptor.getOperands();
  unsigned srcElems = getTotalElemsPerThread(types[0]);
  SmallVector<SmallVector<Value>> srcValues(srcElems);
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    auto values = converter.unpackLLElements(loc, operands[i], rewriter, types[i]);

    assert(values.size() == srcValues.size());
    for (unsigned j = 0; j < srcValues.size(); ++j) {
      srcValues[j].push_back(values[j]);
    }
  }
  return srcValues;
}

// Lowering using warp shuffle operations to do warp level scan.
LogicalResult
ScanOpConversion::emitFastScan(triton::ScanOp op, triton::ScanOpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  ScanLoweringHelper helper(op);
  auto loc = helper.getLoc();
  if (!helper.isSupported())
    return failure();

  Value threadId = getThreadId(rewriter, loc);
  auto mod = op->getParentOfType<ModuleOp>();
  unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  Value warpSize = i32_val(iWarpSize);
  Value warpId = udiv(threadId, warpSize);
  Value laneId = urem(threadId, warpSize);

  auto [laneIdAxis, warpIdAxis, flatIdParallel] =
      getDelinearizedIds(rewriter, helper, laneId, warpId);
  // auto input = adaptor.getOperands()[0];
  // auto type = cast<RankedTensorType>(op.getOperand(0).getType());
  auto axisNumWarps = helper.getAxisNumWarpsWithUniqueData();
  auto axisNumThreads = helper.getAxisNumThreadsPerWarp();
  warpIdAxis = urem(warpIdAxis, i32_val(axisNumWarps));
  auto srcValues =
     unpackInputs(loc, op, adaptor, rewriter, *getTypeConverter());

  // Scan contigous elements in a thread and update `srcValues`.
  scanThreadContiguousElements(srcValues, rewriter, helper);
  // Apply warp level scan to the last element of each chunk of contiguous
  // elements.
  warpScan(srcValues, rewriter, helper, laneIdAxis);

  if (axisNumWarps > 1) {
    // Slow path for the case where there are multiple warps with unique data on
    // the axis.
    // Type elemPtrTys = spirv::PointerType::get(srcValues[0].getType(),
    //                                           spirv::StorageClass::Workgroup);
    // Value baseSharedMemPtr = bitcast(
    //     getSharedMemoryBase(loc, rewriter, op.getOperation()), elemPtrTys);
    auto elems = helper.getScratchSizeInElems();
    SmallVector<Value> smemBases = getSmemBases(helper, op, elems, rewriter);
    SmallVector<Type> smemTypes(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      smemTypes[i] = getElementType(op, i);
    }
    // Store the partial reducing for each warp into shared memory.
    storeWarpAccumulator(srcValues, rewriter, helper, laneIdAxis, warpIdAxis,
                         smemBases, smemTypes, flatIdParallel);
    barrier();
    // Read back the partial reduction of each warp and accumulate them based on
    // warpId. Then update each chunk of contiguous elements by adding the
    // accumulated value from the previous lane.
    AddPartialReduce(srcValues, rewriter, helper, smemBases, smemTypes, warpIdAxis,
                     laneIdAxis, flatIdParallel);
  } else if (srcValues.size() > 1) {
    // Fast path for the case where there is only one warp with unique data on
    // the axis.
    unsigned scanDim = helper.getAxisNumThreadsPerWarpWithUniqueData();
    auto multiDimLaneId = getMultiDimLaneId(rewriter, helper, laneId);
    multiDimLaneId[helper.getAxis()] = i32_val(scanDim - 1);
    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(helper.getEncoding());
    auto laneIdLast = linearize(rewriter, loc, multiDimLaneId, threadsPerWarp,
                                triton::gpu::getOrder(helper.getEncoding()));
    AddPartialReduceOneWarp(srcValues, rewriter, helper, warpIdAxis, laneIdAxis,
                            laneIdLast);
  } // else axisNumWarps == 1 and srcValues.size() == 1, nothing to do.

  // Value results = getTypeConverter()->packLLElements(loc, srcValues, rewriter,
  //                                                    input.getType());
  auto transpose = [](const SmallVector<SmallVector<Value>> &v) {
    assert(v.size() > 0 && v[0].size() > 0);
    auto ret = SmallVector<SmallVector<Value>>(v[0].size(), SmallVector<Value>(v.size()));
    for (int i = 0; i < v.size(); ++i) {
      for (int j = 0; j < v[0].size(); ++j) {
        ret[j][i] = v[i][j];
      }
    }
    return ret;
  };

  SmallVector<Value> results(op.getNumOperands());
  auto valuesTransposed = transpose(srcValues);
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult()[i].getType());
    results[i] = converter -> packLLElements(loc, valuesTransposed[i], rewriter, resultTy);
  }
  rewriter.replaceOp(op, results);
  return success();
}
} // namespace

void populateScanOpToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<ScanOpConversion>(typeConverter, context, allocation,
                                 indexCacheInfo, benefit);
}
