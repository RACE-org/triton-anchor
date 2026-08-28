#include "triton-anchor/Validation/AnchorIRValidator.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

namespace mlir::triton::anchor {
namespace {

// The Validator reports complete paths for every inspected object.  Keeping
// traversal below these limits prevents a deliberately deep Region tree or a
// wide invalid module from turning those paths and diagnostics into a
// quadratic memory allocation before the normal verifier is reached.
constexpr unsigned kMaxAnchorIRStructuralDepth = 256;
constexpr size_t kMaxAnchorIROperations = 16384;
constexpr size_t kMaxAnchorIRDiagnostics = 8192;
// This is deliberately a logical-visit budget rather than a unique-node
// budget.  Diagnostics are path-sensitive, so memoizing a shared Type or
// Attribute would silently omit a second illegal occurrence.  The budget
// instead makes exponentially shared DAGs fail closed while retaining all
// diagnostics up to the bounded point.
constexpr size_t kMaxAnchorIRObjectVisits = 262144;

// Untrusted symbol names, locations and printed objects are repeated across
// diagnostics. Keep each public field bounded so a compact IR cannot amplify
// one long string through the complete diagnostic budget.
constexpr size_t kMaxAnchorIRDiagnosticFieldBytes = 1024;
constexpr llvm::StringLiteral kTruncatedFieldMarker = "...<truncated>...";

// Text parsing temporarily changes MLIRContext::allowUnregisteredDialects and
// installs a context-local diagnostic handler. Calls sharing one explicit
// context must therefore be serialized across both validate and normalize.
// Weak entries avoid retaining contexts after their callers release them.
std::shared_ptr<std::recursive_mutex>
getAnchorIRTextContextMutex(MLIRContext &context) {
  static std::mutex registryMutex;
  static std::unordered_map<MLIRContext *, std::weak_ptr<std::recursive_mutex>>
      contextMutexes;

  std::lock_guard<std::mutex> registryGuard(registryMutex);
  for (auto iterator = contextMutexes.begin();
       iterator != contextMutexes.end();) {
    if (iterator->second.expired())
      iterator = contextMutexes.erase(iterator);
    else
      ++iterator;
  }

  std::weak_ptr<std::recursive_mutex> &entry = contextMutexes[&context];
  if (std::shared_ptr<std::recursive_mutex> existing = entry.lock())
    return existing;
  auto created = std::make_shared<std::recursive_mutex>();
  entry = created;
  return created;
}

std::string boundDiagnosticField(std::string value) {
  if (value.size() <= kMaxAnchorIRDiagnosticFieldBytes)
    return value;
  size_t remaining =
      kMaxAnchorIRDiagnosticFieldBytes - kTruncatedFieldMarker.size();
  size_t prefixSize = remaining / 2;
  while (prefixSize != 0 &&
         !llvm::json::isUTF8(llvm::StringRef(value.data(), prefixSize)))
    --prefixSize;
  size_t suffixStart = value.size() - (remaining - prefixSize);
  while (suffixStart < value.size() &&
         (static_cast<unsigned char>(value[suffixStart]) & 0xC0) == 0x80)
    ++suffixStart;
  return value.substr(0, prefixSize) + kTruncatedFieldMarker.str() +
         value.substr(suffixStart);
}

struct CapturedMLIRDiagnostic {
  std::string message;
  SourceLocation location;
};

struct DiagnosticCaptureResult {
  LogicalResult result;
  std::optional<CapturedMLIRDiagnostic> error;
};

std::string sanitizeUTF8(llvm::StringRef value) {
  std::string sanitized;
  while (!value.empty()) {
    size_t invalidOffset = 0;
    if (llvm::json::isUTF8(value, &invalidOffset)) {
      sanitized.append(value.data(), value.size());
      break;
    }
    sanitized.append(value.data(), invalidOffset);
    unsigned char invalid = value[invalidOffset];
    constexpr char hex[] = "0123456789ABCDEF";
    sanitized.push_back('%');
    sanitized.push_back(hex[invalid >> 4]);
    sanitized.push_back(hex[invalid & 0x0F]);
    value = value.drop_front(invalidOffset + 1);
  }
  return boundDiagnosticField(std::move(sanitized));
}

SourceLocation getSourceLocation(Location location) {
  SourceLocation result;
  auto fileLoc =
      static_cast<LocationAttr>(location).findInstanceOf<FileLineColLoc>();
  if (!fileLoc)
    return result;
  result.valid = true;
  result.file = sanitizeUTF8(fileLoc.getFilename());
  result.line = fileLoc.getLine();
  result.column = fileLoc.getColumn();
  return result;
}

std::string replaceAll(std::string value, llvm::StringRef token,
                       llvm::StringRef replacement) {
  size_t position = 0;
  while ((position = value.find(token.str(), position)) != std::string::npos) {
    value.replace(position, token.size(), replacement.str());
    position += replacement.size();
  }
  return value;
}

std::string escapePathComponent(llvm::StringRef value) {
  std::string escaped;
  escaped.reserve(value.size());
  constexpr char hex[] = "0123456789ABCDEF";
  for (unsigned char character : value.bytes()) {
    bool safe = llvm::isAlnum(character) || character == '_' ||
                character == '-' || character == '.';
    if (safe) {
      escaped.push_back(static_cast<char>(character));
      continue;
    }
    escaped.push_back('%');
    escaped.push_back(hex[character >> 4]);
    escaped.push_back(hex[character & 0x0F]);
  }
  return boundDiagnosticField(std::move(escaped));
}

std::string render(llvm::StringRef value, llvm::StringRef dialect,
                   llvm::StringRef operation, llvm::StringRef object) {
  std::string rendered = value.str();
  rendered = replaceAll(std::move(rendered), "{dialect}", dialect);
  rendered = replaceAll(std::move(rendered), "{operation}", operation);
  return replaceAll(std::move(rendered), "{object}", object);
}

Diagnostic makeTemplateDiagnostic(
    const ValidationPolicy &policy,
    const DiagnosticTemplate &diagnosticTemplate, llvm::StringRef objectKind,
    llvm::StringRef objectName, llvm::StringRef operationPath,
    SourceLocation location, llvm::StringRef dialect = {},
    llvm::StringRef objectPath = {}, llvm::StringRef renderOperation = {}) {
  std::string safeObjectName = sanitizeUTF8(objectName);
  std::string safeOperationPath = sanitizeUTF8(operationPath);
  std::string safeObjectPath = sanitizeUTF8(objectPath);
  std::string safeDialect = sanitizeUTF8(dialect);
  std::string safeRenderOperation =
      sanitizeUTF8(renderOperation.empty() ? objectName : renderOperation);
  Diagnostic diagnostic;
  diagnostic.code = diagnosticTemplate.code;
  diagnostic.message =
      sanitizeUTF8(render(diagnosticTemplate.message, safeDialect,
                          safeRenderOperation, safeObjectName));
  diagnostic.hint = sanitizeUTF8(render(diagnosticTemplate.hint, safeDialect,
                                        safeRenderOperation, safeObjectName));
  diagnostic.specVersion = policy.specVersion;
  diagnostic.track = policy.track;
  diagnostic.phase = policy.phase;
  diagnostic.objectKind = objectKind.str();
  diagnostic.objectName = std::move(safeObjectName);
  diagnostic.operationPath = std::move(safeOperationPath);
  diagnostic.objectPath = std::move(safeObjectPath);
  diagnostic.location = std::move(location);
  return diagnostic;
}

void appendResourceLimitDiagnostic(const ValidationPolicy &policy,
                                   ValidationReport &report,
                                   llvm::StringRef objectName,
                                   llvm::StringRef operationPath,
                                   SourceLocation location = {}) {
  if (report.resourceLimitReported)
    return;
  report.resourceLimitReported = true;
  report.diagnostics.push_back(
      makeTemplateDiagnostic(policy, policy.resourceLimit, "module", objectName,
                             operationPath, std::move(location)));
}

void appendDiagnostic(const ValidationPolicy &policy, ValidationReport &report,
                      Diagnostic diagnostic) {
  if (report.resourceLimitReported)
    return;
  // Reserve the final slot for a stable explanation instead of silently
  // truncating an otherwise unbounded collection of recursive diagnostics.
  if (report.diagnostics.size() < kMaxAnchorIRDiagnostics - 1) {
    report.diagnostics.push_back(std::move(diagnostic));
    return;
  }
  appendResourceLimitDiagnostic(policy, report, "diagnostic count",
                                "builtin.module");
}

ValidationReport makeResourceLimitReport(const ValidationPolicy &policy,
                                         llvm::StringRef objectName,
                                         SourceLocation location = {}) {
  ValidationReport report{policy.specVersion, policy.track, policy.phase, {}};
  appendResourceLimitDiagnostic(policy, report, objectName, "builtin.module",
                                std::move(location));
  return report;
}

bool hasInvariant(const ValidationPolicy &policy, llvm::StringRef name) {
  return policy.enabledInvariants.count(name.str());
}

void diagnoseSemantic(const ValidationPolicy &policy, llvm::StringRef invariant,
                      llvm::StringRef objectKind, llvm::StringRef objectName,
                      llvm::StringRef operationPath, Location location,
                      ValidationReport &report, llvm::StringRef objectPath = {},
                      llvm::StringRef renderOperation = {}) {
  auto diagnostic = policy.semanticDiagnostics.find(invariant.str());
  if (diagnostic == policy.semanticDiagnostics.end())
    return;
  appendDiagnostic(policy, report,
                   makeTemplateDiagnostic(policy, diagnostic->second,
                                          objectKind, objectName, operationPath,
                                          getSourceLocation(location), {},
                                          objectPath, renderOperation));
}

template <typename T> std::string printIRObject(T object) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  object.print(stream);
  return result;
}

std::string printNormalizedModule(ModuleOp module) {
  OwningOpRef<ModuleOp> normalizedModule = module.clone();
  AttrTypeReplacer locationReplacer;
  locationReplacer.addReplacement(
      [context = module.getContext()](LocationAttr) -> Attribute {
        return UnknownLoc::get(context);
      });
  locationReplacer.recursivelyReplaceElementsIn(
      normalizedModule->getOperation(),
      /*replaceAttrs=*/true,
      /*replaceLocs=*/true,
      /*replaceTypes=*/true);

  auto printModule = [&](bool useLocalScope) {
    std::string output;
    llvm::raw_string_ostream outputStream(output);
    OpPrintingFlags flags;
    flags.printGenericOpForm().assumeVerified().enableDebugInfo(false);
    if (useLocalScope)
      flags.useLocalScope();
    AsmState state(normalizedModule->getOperation(), flags);
    normalizedModule->print(outputStream, state);
    outputStream.flush();
    bool referencedResources = !state.getDialectResources().empty();
    return std::make_pair(std::move(output), referencedResources);
  };

  // Preserve the established local-scope canonical form for ordinary IR.  A
  // local-scope print still records every dialect resource handle it visits,
  // but MLIR intentionally omits the resource payload section from its text.
  // If such a handle was referenced, repeat as a root print so semantic bytes
  // (including DenseResourceElementsAttr payloads) enter the normalized text
  // and SHA-256.  This keeps existing resource-free Golden files byte-stable.
  auto [printed, referencedResources] = printModule(/*useLocalScope=*/true);
  if (referencedResources)
    printed = printModule(/*useLocalScope=*/false).first;

  std::string normalized;
  normalized.reserve(printed.size() + 1);
  for (size_t index = 0; index < printed.size(); ++index) {
    if (printed[index] != '\r') {
      normalized.push_back(printed[index]);
      continue;
    }
    if (index + 1 < printed.size() && printed[index + 1] == '\n')
      ++index;
    normalized.push_back('\n');
  }
  while (!normalized.empty() && normalized.back() == '\n')
    normalized.pop_back();
  normalized.push_back('\n');
  return normalized;
}

std::optional<unsigned> getTritonGPUEncodingRank(Attribute encoding) {
  if (auto dot = dyn_cast<triton::gpu::DotOperandEncodingAttr>(encoding))
    return getTritonGPUEncodingRank(dot.getParent());
  if (auto slice = dyn_cast<triton::gpu::SliceEncodingAttr>(encoding)) {
    auto parentRank = getTritonGPUEncodingRank(slice.getParent());
    if (!parentRank || *parentRank == 0 || slice.getDim() >= *parentRank)
      return std::nullopt;
    return *parentRank - 1;
  }
  if (auto shared = dyn_cast<triton::gpu::SharedEncodingAttr>(encoding))
    return shared.getOrder().size();
  if (auto distributed =
          dyn_cast<triton::gpu::DistributedEncodingTrait>(encoding))
    return distributed.getWarpsPerCTA().size();
  return std::nullopt;
}

bool isPositivePowerOfTwo(unsigned value) {
  return value != 0 && (value & (value - 1)) == 0;
}

bool isPositivePowerOfTwo(int64_t value) {
  return value > 0 && (static_cast<uint64_t>(value) &
                       (static_cast<uint64_t>(value) - 1)) == 0;
}

bool allPositivePowersOfTwo(llvm::ArrayRef<unsigned> values) {
  return !values.empty() && llvm::all_of(values, [](unsigned value) {
    return isPositivePowerOfTwo(value);
  });
}

bool allPositive(llvm::ArrayRef<unsigned> values) {
  return !values.empty() &&
         llvm::all_of(values, [](unsigned value) { return value != 0; });
}

bool isPermutationOfRank(llvm::ArrayRef<unsigned> order, unsigned rank) {
  if (order.size() != rank)
    return false;
  llvm::SmallVector<bool> seen(rank, false);
  for (unsigned value : order) {
    if (value >= rank || seen[value])
      return false;
    seen[value] = true;
  }
  return true;
}

bool isPermutationOfRank(llvm::ArrayRef<int32_t> order, unsigned rank) {
  if (order.size() != rank)
    return false;
  llvm::SmallVector<bool> seen(rank, false);
  for (int32_t value : order) {
    if (value < 0 || static_cast<uint64_t>(value) >= rank ||
        seen[static_cast<unsigned>(value)])
      return false;
    seen[static_cast<unsigned>(value)] = true;
  }
  return true;
}

std::optional<unsigned> getNonNegativeUnsigned(IntegerAttr attribute) {
  if (!attribute || attribute.getValue().isNegative() ||
      attribute.getValue().getActiveBits() >
          std::numeric_limits<unsigned>::digits)
    return std::nullopt;
  return static_cast<unsigned>(attribute.getValue().getZExtValue());
}

std::optional<unsigned> getTensorPointerRank(Type type) {
  auto pointer = dyn_cast<triton::PointerType>(type);
  if (!pointer)
    return std::nullopt;
  auto tensor = dyn_cast<RankedTensorType>(pointer.getPointeeType());
  if (!tensor)
    return std::nullopt;
  return static_cast<unsigned>(tensor.getRank());
}

bool hasValidBoundaryCheck(Operation *operation, unsigned pointerOperandIndex) {
  auto boundary = operation->getAttrOfType<DenseI32ArrayAttr>("boundaryCheck");
  bool hasPadding = operation->hasAttr("padding");
  if ((!boundary || boundary.empty()) && !hasPadding)
    return true;
  if (pointerOperandIndex >= operation->getNumOperands())
    return false;

  std::optional<unsigned> rank = getTensorPointerRank(
      operation->getOperand(pointerOperandIndex).getType());
  if (!rank)
    return false;

  llvm::SmallVector<bool> seen(*rank, false);
  if (boundary) {
    for (int32_t dimension : boundary.asArrayRef()) {
      if (dimension < 0 || static_cast<uint64_t>(dimension) >= *rank ||
          seen[static_cast<unsigned>(dimension)])
        return false;
      seen[static_cast<unsigned>(dimension)] = true;
    }
  }
  return true;
}

bool hasSafeCTALayout(triton::gpu::CTALayoutAttr layout, unsigned rank) {
  if (!layout)
    return false;
  llvm::ArrayRef<unsigned> ctas = layout.getCTAsPerCGA();
  llvm::ArrayRef<unsigned> splits = layout.getCTASplitNum();
  llvm::ArrayRef<unsigned> order = layout.getCTAOrder();
  if (ctas.size() != rank || splits.size() != rank ||
      !allPositivePowersOfTwo(ctas) || !allPositivePowersOfTwo(splits) ||
      !isPermutationOfRank(order, rank))
    return false;
  for (auto [cta, split] : llvm::zip(ctas, splits))
    if (cta % split != 0)
      return false;
  return true;
}

bool isSupportedGPUScalarType(Type type) {
  return type.isSignlessInteger(1) || type.isSignlessInteger(8) ||
         type.isSignlessInteger(16) || type.isSignlessInteger(32) ||
         type.isSignlessInteger(64) || type.isFloat8E4M3FNUZ() ||
         type.isFloat8E5M2() || type.isFloat8E5M2FNUZ() || type.isF16() ||
         type.isBF16() || type.isF32() || type.isF64();
}

bool hasSupportedGPUElementType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return isSupportedGPUScalarType(tensor.getElementType()) ||
           isa<triton::PointerType>(tensor.getElementType());
  if (auto memdesc = dyn_cast<triton::MemDescType>(type))
    return isSupportedGPUScalarType(memdesc.getElementType());
  return true;
}

bool hasSafeGPUShape(llvm::ArrayRef<int64_t> shape,
                     bool enforceTensorElementContract) {
  if (shape.empty())
    return false;
  uint64_t elements = 1;
  // Keep this in sync with triton/Dialect/Triton/IR/Traits.h.  The upstream
  // verifier multiplies in int64_t and only runs for Triton operations, so a
  // function-signature-only type needs this checked calculation here.
  constexpr uint64_t maxTritonTensorElements = 1048576;
  for (int64_t value : shape) {
    if (value <= 0)
      return false;
    if (!enforceTensorElementContract)
      continue;
    uint64_t dimension = static_cast<uint64_t>(value);
    if (elements > maxTritonTensorElements / dimension)
      return false;
    elements *= dimension;
  }
  return !enforceTensorElementContract ||
         (elements != 0 && (elements & (elements - 1)) == 0);
}

bool hasSafeBlockedEncoding(triton::gpu::BlockedEncodingAttr blocked,
                            llvm::ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  llvm::ArrayRef<unsigned> sizePerThread = blocked.getSizePerThread__();
  llvm::ArrayRef<unsigned> threadsPerWarp = blocked.getThreadsPerWarp__();
  llvm::ArrayRef<unsigned> warpsPerCTA = blocked.getWarpsPerCTA__();
  llvm::ArrayRef<unsigned> order = blocked.getOrder();
  if (sizePerThread.size() != rank || threadsPerWarp.size() != rank ||
      warpsPerCTA.size() != rank || !allPositivePowersOfTwo(sizePerThread) ||
      !allPositivePowersOfTwo(threadsPerWarp) ||
      !allPositivePowersOfTwo(warpsPerCTA) ||
      !isPermutationOfRank(order, rank) ||
      !hasSafeCTALayout(blocked.getCTALayout(), rank))
    return false;

  for (unsigned index = 0; index < rank; ++index) {
    uint64_t tile = sizePerThread[index];
    constexpr uint64_t maxTile = std::numeric_limits<unsigned>::max();
    if (tile > maxTile / threadsPerWarp[index])
      return false;
    tile *= threadsPerWarp[index];
    if (tile > maxTile / warpsPerCTA[index])
      return false;
    tile *= warpsPerCTA[index];
  }
  return true;
}

bool hasSafeSharedEncoding(triton::gpu::SharedEncodingAttr shared,
                           llvm::ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  llvm::ArrayRef<unsigned> order = shared.getOrder();
  if (!isPositivePowerOfTwo(shared.getVec()) ||
      !isPositivePowerOfTwo(shared.getPerPhase()) ||
      !isPositivePowerOfTwo(shared.getMaxPhase()) ||
      !isPermutationOfRank(order, rank) ||
      !hasSafeCTALayout(shared.getCTALayout(), rank))
    return false;

  uint64_t swizzleWidth =
      static_cast<uint64_t>(shared.getVec()) * shared.getMaxPhase();
  if (shared.getMaxPhase() != 1 &&
      swizzleWidth > static_cast<uint64_t>(shape[order.front()]))
    return false;
  if (shared.getHasLeadingOffset()) {
    auto swizzle = std::make_pair(shared.getPerPhase(), shared.getMaxPhase());
    if (swizzle != std::make_pair(4u, 2u) &&
        swizzle != std::make_pair(2u, 4u) && swizzle != std::make_pair(1u, 8u))
      return false;
  }
  return true;
}

bool hasSafeNvidiaMmaEncoding(triton::gpu::NvidiaMmaEncodingAttr mma,
                              llvm::ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  llvm::ArrayRef<unsigned> warps = mma.getWarpsPerCTA__();
  llvm::ArrayRef<unsigned> instruction = mma.getInstrShape();
  if (warps.size() != rank || !allPositive(warps) ||
      !hasSafeCTALayout(mma.getCTALayout(), rank))
    return false;

  switch (mma.getVersionMajor()) {
  case 1:
    return rank == 2 && mma.getVersionMinor() < 512 &&
           instruction == llvm::ArrayRef<unsigned>({16, 16});
  case 2:
    if (mma.getVersionMinor() > 1)
      return false;
    if (rank == 2)
      return instruction == llvm::ArrayRef<unsigned>({16, 8});
    return rank == 3 && instruction == llvm::ArrayRef<unsigned>({1, 16, 8});
  case 3: {
    if (rank != 2 || mma.getVersionMinor() != 0 || instruction.size() != 3 ||
        instruction[0] != 16 ||
        (instruction[1] != 16 && instruction[1] != 32 && instruction[1] != 64 &&
         instruction[1] != 128 && instruction[1] != 256) ||
        (instruction[2] != 8 && instruction[2] != 16 && instruction[2] != 32))
      return false;
    uint64_t warpCount = 1;
    for (unsigned value : warps) {
      if (warpCount > std::numeric_limits<uint64_t>::max() / value)
        return false;
      warpCount *= value;
    }
    return warpCount % 4 == 0;
  }
  default:
    return false;
  }
}

bool hasSafeAMDMfmaEncoding(triton::gpu::AMDMfmaEncodingAttr mma,
                            llvm::ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  llvm::ArrayRef<unsigned> warps = mma.getWarpsPerCTA__();
  bool instructionSupported = (mma.getMDim() == 16 && mma.getNDim() == 16) ||
                              (mma.getMDim() == 32 && mma.getNDim() == 32);
  return (rank == 2 || rank == 3) && mma.getVersionMajor() >= 1 &&
         mma.getVersionMajor() <= 3 && mma.getVersionMinor() == 0 &&
         instructionSupported && warps.size() == rank && allPositive(warps) &&
         hasSafeCTALayout(mma.getCTALayout(), rank);
}

bool hasSafeAMDWmmaEncoding(triton::gpu::AMDWmmaEncodingAttr mma,
                            llvm::ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  llvm::ArrayRef<unsigned> warps = mma.getWarpsPerCTA__();
  return (rank == 2 || rank == 3) && warps.size() == rank &&
         allPositive(warps) && hasSafeCTALayout(mma.getCTALayout(), rank);
}

#if defined(TRITON_ANCHOR_HAS_FANT_WMMA)
bool hasSafeFANTWmmaEncoding(triton::gpu::FANTWmmaEncodingAttr mma,
                             llvm::ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  llvm::ArrayRef<unsigned> warps = mma.getWarpsPerCTA__();
  return (rank == 2 || rank == 3) && warps.size() == rank &&
         allPositive(warps) && hasSafeCTALayout(mma.getCTALayout(), rank);
}
#endif

std::optional<llvm::SmallVector<unsigned>>
getSafeCoreCTAsPerCGA(Attribute encoding) {
  auto getCTAs = [](triton::gpu::CTALayoutAttr layout)
      -> std::optional<llvm::SmallVector<unsigned>> {
    if (!layout)
      return std::nullopt;
    return llvm::SmallVector<unsigned>(layout.getCTAsPerCGA());
  };
  if (auto blocked = dyn_cast<triton::gpu::BlockedEncodingAttr>(encoding))
    return getCTAs(blocked.getCTALayout());
  if (auto mma = dyn_cast<triton::gpu::NvidiaMmaEncodingAttr>(encoding))
    return getCTAs(mma.getCTALayout());
  if (auto mma = dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(encoding))
    return getCTAs(mma.getCTALayout());
  if (auto mma = dyn_cast<triton::gpu::AMDWmmaEncodingAttr>(encoding))
    return getCTAs(mma.getCTALayout());
#if defined(TRITON_ANCHOR_HAS_FANT_WMMA)
  if (auto mma = dyn_cast<triton::gpu::FANTWmmaEncodingAttr>(encoding))
    return getCTAs(mma.getCTALayout());
#endif
  if (auto slice = dyn_cast<triton::gpu::SliceEncodingAttr>(encoding)) {
    auto parentCTAs = getSafeCoreCTAsPerCGA(slice.getParent());
    if (!parentCTAs || slice.getDim() >= parentCTAs->size() ||
        (*parentCTAs)[slice.getDim()] != 1)
      return std::nullopt;
    parentCTAs->erase(parentCTAs->begin() + slice.getDim());
    return parentCTAs;
  }
  return std::nullopt;
}

bool hasSafeCoreTritonGPUEncoding(Attribute encoding,
                                  llvm::ArrayRef<int64_t> shape,
                                  bool enforceTensorElementContract = true) {
  if (!encoding || !hasSafeGPUShape(shape, enforceTensorElementContract))
    return false;
  if (auto blocked = dyn_cast<triton::gpu::BlockedEncodingAttr>(encoding))
    return hasSafeBlockedEncoding(blocked, shape);
  if (auto shared = dyn_cast<triton::gpu::SharedEncodingAttr>(encoding))
    return hasSafeSharedEncoding(shared, shape);
  if (auto mma = dyn_cast<triton::gpu::NvidiaMmaEncodingAttr>(encoding))
    return hasSafeNvidiaMmaEncoding(mma, shape);
  if (auto mma = dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(encoding))
    return hasSafeAMDMfmaEncoding(mma, shape);
  if (auto mma = dyn_cast<triton::gpu::AMDWmmaEncodingAttr>(encoding))
    return hasSafeAMDWmmaEncoding(mma, shape);
#if defined(TRITON_ANCHOR_HAS_FANT_WMMA)
  if (auto mma = dyn_cast<triton::gpu::FANTWmmaEncodingAttr>(encoding))
    return hasSafeFANTWmmaEncoding(mma, shape);
#endif
  if (auto dot = dyn_cast<triton::gpu::DotOperandEncodingAttr>(encoding)) {
    Attribute parent = dot.getParent();
    bool kWidthSupported = false;
    if (auto mma = dyn_cast_or_null<triton::gpu::NvidiaMmaEncodingAttr>(parent))
      kWidthSupported = mma.getVersionMajor() == 2 ? dot.getKWidth() != 0
                                                   : dot.getKWidth() == 0;
    else if (isa_and_nonnull<triton::gpu::AMDWmmaEncodingAttr>(parent))
      kWidthSupported = dot.getKWidth() == 16;
    else if (isa_and_nonnull<triton::gpu::AMDMfmaEncodingAttr>(parent))
      kWidthSupported = dot.getKWidth() != 0;
#if defined(TRITON_ANCHOR_HAS_FANT_WMMA)
    else if (isa_and_nonnull<triton::gpu::FANTWmmaEncodingAttr>(parent))
      kWidthSupported =
          dot.getKWidth() == 2 || dot.getKWidth() == 4 || dot.getKWidth() == 8;
#endif
    else if (isa_and_nonnull<triton::gpu::BlockedEncodingAttr>(parent))
      kWidthSupported = dot.getKWidth() == 0;
    return dot.getOpIdx() <= 1 && kWidthSupported &&
           hasSafeCoreTritonGPUEncoding(parent, shape,
                                        enforceTensorElementContract);
  }
  if (auto slice = dyn_cast<triton::gpu::SliceEncodingAttr>(encoding)) {
    Attribute parent = slice.getParent();
    bool parentKindSupported = isa_and_nonnull<
        triton::gpu::BlockedEncodingAttr, triton::gpu::NvidiaMmaEncodingAttr,
        triton::gpu::AMDMfmaEncodingAttr, triton::gpu::AMDWmmaEncodingAttr,
        triton::gpu::SliceEncodingAttr>(parent);
#if defined(TRITON_ANCHOR_HAS_FANT_WMMA)
    parentKindSupported =
        parentKindSupported ||
        isa_and_nonnull<triton::gpu::FANTWmmaEncodingAttr>(parent);
#endif
    auto parentRank = getTritonGPUEncodingRank(parent);
    auto parentCTAs = getSafeCoreCTAsPerCGA(parent);
    if (!parentKindSupported || !parentRank ||
        *parentRank != shape.size() + 1 || slice.getDim() >= *parentRank ||
        !parentCTAs || slice.getDim() >= parentCTAs->size() ||
        (*parentCTAs)[slice.getDim()] != 1)
      return false;
    llvm::SmallVector<int64_t> parentShape(shape);
    parentShape.insert(parentShape.begin() + slice.getDim(), 1);
    return hasSafeCoreTritonGPUEncoding(parent, parentShape,
                                        enforceTensorElementContract);
  }
  return false;
}

bool productEquals(llvm::ArrayRef<unsigned> values, unsigned expected) {
  uint64_t result = 1;
  for (unsigned value : values) {
    if (value == 0 || result > expected / value)
      return false;
    result *= value;
  }
  return result == expected;
}

Attribute getTritonGPUConfigurationEncoding(Attribute encoding) {
  // Dot-operand and slice encodings derive their execution topology from a
  // parent layout.  Calling every DistributedEncodingTrait method directly on
  // a dot operand is not safe in all supported Triton revisions and can abort
  // on otherwise diagnosable IR, so unwrap these wrappers first.
  while (encoding) {
    if (auto dot = dyn_cast<triton::gpu::DotOperandEncodingAttr>(encoding)) {
      encoding = dot.getParent();
      continue;
    }
    if (auto slice = dyn_cast<triton::gpu::SliceEncodingAttr>(encoding)) {
      encoding = slice.getParent();
      continue;
    }
    break;
  }
  return encoding;
}

std::optional<int> getPositiveIntegerAttribute(ModuleOp module,
                                               llvm::StringRef name);
llvm::StringRef getDialectNamespace(Attribute attribute);

Attribute getTritonGPUShapedEncoding(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.getEncoding();
  if (auto memdesc = dyn_cast<triton::MemDescType>(type))
    return memdesc.getEncoding();
  return {};
}

std::optional<llvm::ArrayRef<int64_t>> getTritonGPUShapedShape(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.getShape();
  if (auto memdesc = dyn_cast<triton::MemDescType>(type))
    return memdesc.getShape();
  return std::nullopt;
}

void validateTritonGPUShapedType(Type type, llvm::StringRef operationPath,
                                 llvm::StringRef objectPath, Location location,
                                 ModuleOp module,
                                 const ValidationPolicy &policy,
                                 ValidationReport &report) {
  if (!hasInvariant(policy, "gpu.tensor_encoding"))
    return;
  std::optional<llvm::ArrayRef<int64_t>> shape = getTritonGPUShapedShape(type);
  if (!shape)
    return;
  Attribute encoding = getTritonGPUShapedEncoding(type);
  llvm::ArrayRef<int64_t> encodingShape = *shape;
  // Triton's software-pipelining pass intentionally represents staged shared
  // memory as a MemDesc whose leading dimension is numStages while the Shared
  // layout remains one rank smaller.  This is an upstream contract, not a
  // malformed rank.  Keep the exception limited to Shared MemDesc and require
  // a positive stage count; ordinary tensors retain exact-rank validation.
  if (isa<triton::MemDescType>(type)) {
    if (auto shared =
            dyn_cast_or_null<triton::gpu::SharedEncodingAttr>(encoding)) {
      if (shape->size() == shared.getOrder().size() + 1 && shape->front() > 0)
        encodingShape = shape->drop_front();
    }
  }
  llvm::StringRef encodingDialect =
      encoding ? getDialectNamespace(encoding) : llvm::StringRef();
  if (!encoding || encodingDialect.empty() || encodingDialect == "builtin") {
    diagnoseSemantic(policy, "gpu.tensor_encoding", "type", objectPath,
                     operationPath, location, report, objectPath,
                     operationPath);
    return;
  }

  bool coreEncoding = encodingDialect == "triton_gpu";
  bool extensionEncoding =
      policy.extensionDialects.count(encodingDialect.str());
  if (!coreEncoding && !extensionEncoding) {
    // An allowed operation/attribute namespace is not automatically a tensor
    // layout namespace.  Only the Track's triton_gpu layouts, or a backend
    // encoding dialect explicitly declared for post-hook validation, satisfy
    // the shaped-type contract.
    if (policy.allowedDialects.count(encodingDialect.str()))
      diagnoseSemantic(policy, "gpu.tensor_encoding", "type", objectPath,
                       operationPath, location, report, objectPath,
                       operationPath);
    return;
  }

  if (hasInvariant(policy, "gpu.shaped_element_type") &&
      !hasSupportedGPUElementType(type)) {
    diagnoseSemantic(policy, "gpu.shaped_element_type", "type", objectPath,
                     operationPath, location, report,
                     (llvm::Twine(objectPath) + ".element_type").str(),
                     operationPath);
  }

  // Vendor encodings explicitly admitted during post-hook validation remain an
  // extension responsibility.  Core TritonGPU encodings, however, have a
  // structural rank that this validator can and must check.
  bool encodingComponentsSafe = true;
  bool coreEncodingKindSafe = true;
  if (coreEncoding) {
    if (isa<RankedTensorType>(type))
      coreEncodingKindSafe = !isa<triton::gpu::SharedEncodingAttr>(encoding);
    else if (isa<triton::MemDescType>(type))
      coreEncodingKindSafe = isa<triton::gpu::SharedEncodingAttr>(encoding);
    if (!coreEncodingKindSafe &&
        hasInvariant(policy, "gpu.encoding_components"))
      diagnoseSemantic(policy, "gpu.encoding_components", "attribute",
                       printIRObject(encoding), operationPath, location, report,
                       (llvm::Twine(objectPath) + ".encoding").str(),
                       operationPath);
  }
  if (hasInvariant(policy, "gpu.encoding_rank") && coreEncoding) {
    auto encodingRank = getTritonGPUEncodingRank(encoding);
    if (!encodingRank || *encodingRank != encodingShape.size())
      diagnoseSemantic(policy, "gpu.encoding_rank", "type", objectPath,
                       operationPath, location, report,
                       (llvm::Twine(objectPath) + ".encoding").str(),
                       operationPath);
  }

  if (coreEncoding) {
    encodingComponentsSafe =
        coreEncodingKindSafe &&
        hasSafeCoreTritonGPUEncoding(encoding, encodingShape,
                                     /*enforceTensorElementContract=*/
                                     isa<RankedTensorType>(type));
    if (!encodingComponentsSafe && coreEncodingKindSafe &&
        hasInvariant(policy, "gpu.encoding_components"))
      diagnoseSemantic(policy, "gpu.encoding_components", "attribute",
                       printIRObject(encoding), operationPath, location, report,
                       (llvm::Twine(objectPath) + ".encoding").str(),
                       operationPath);
  }

  if (!hasInvariant(policy, "gpu.module_configuration"))
    return;
  // Never call a DistributedEncodingTrait method until kind-specific support
  // domains have been checked.  Several pinned Triton implementations abort on
  // unknown MMA versions, unsupported ranks, or zero topology components.
  if (!encodingComponentsSafe)
    return;
  auto distributed = dyn_cast_or_null<triton::gpu::DistributedEncodingTrait>(
      getTritonGPUConfigurationEncoding(encoding));
  auto numWarps = getPositiveIntegerAttribute(module, "triton_gpu.num-warps");
  auto threadsPerWarp =
      getPositiveIntegerAttribute(module, "triton_gpu.threads-per-warp");
  auto numCTAs = getPositiveIntegerAttribute(module, "triton_gpu.num-ctas");
  if (!numWarps || !threadsPerWarp || !numCTAs)
    return;
  if (auto shared = dyn_cast<triton::gpu::SharedEncodingAttr>(
          getTritonGPUConfigurationEncoding(encoding))) {
    if (!productEquals(shared.getCTALayout().getCTAsPerCGA(),
                       static_cast<unsigned>(*numCTAs)))
      diagnoseSemantic(policy, "gpu.module_configuration", "type", objectPath,
                       operationPath, location, report,
                       (llvm::Twine(objectPath) + ".encoding").str(),
                       operationPath);
    return;
  }
  if (!distributed)
    return;
  bool matches = productEquals(distributed.getWarpsPerCTA(),
                               static_cast<unsigned>(*numWarps)) &&
                 productEquals(distributed.getThreadsPerWarp(),
                               static_cast<unsigned>(*threadsPerWarp)) &&
                 productEquals(distributed.getCTAsPerCGA(),
                               static_cast<unsigned>(*numCTAs));
  if (!matches)
    diagnoseSemantic(policy, "gpu.module_configuration", "type", objectPath,
                     operationPath, location, report,
                     (llvm::Twine(objectPath) + ".encoding").str(),
                     operationPath);
}

llvm::StringRef getDialectNamespace(Type type) {
  if (auto opaque = dyn_cast<OpaqueType>(type))
    return opaque.getDialectNamespace();
  return type.getDialect().getNamespace();
}

llvm::StringRef getDialectNamespace(Attribute attribute) {
  if (auto opaque = dyn_cast<OpaqueAttr>(attribute))
    return opaque.getDialectNamespace();
  return attribute.getDialect().getNamespace();
}

bool isBuiltinObjectDialectOrAllowed(llvm::StringRef dialect,
                                     const ValidationPolicy &policy) {
  return dialect.empty() || dialect == "builtin" ||
         policy.allowedDialects.count(dialect.str());
}

bool isValidNamedDialectNamespace(llvm::StringRef dialect) {
  return !dialect.empty() && Dialect::isValidNamespace(dialect);
}

struct ObjectTraversalState {
  llvm::SmallPtrSet<const void *, 16> activeTypes;
  llvm::SmallPtrSet<const void *, 16> activeAttributes;
  unsigned depth = 0;
};

void visitType(Type type, llvm::StringRef operationPath,
               llvm::StringRef objectPath, Location location, ModuleOp module,
               const ValidationPolicy &policy, ValidationReport &report,
               ObjectTraversalState &state);

void visitAttribute(Attribute attribute, llvm::StringRef operationPath,
                    llvm::StringRef objectPath, Location location,
                    ModuleOp module, const ValidationPolicy &policy,
                    ValidationReport &report, ObjectTraversalState &state);

// Registered operations can expose generated properties both through
// getAttrs() and through the opaque properties dictionary.  For those ops,
// walk only entries that are not already exposed as an ODS/inherent attribute;
// otherwise function signatures and similar property-backed types are
// diagnosed twice.
//
// Unregistered operations are different: generic syntax may materialize a
// property dictionary whose entries are also queryable through hasAttr(), but
// the properties path is the only faithful description of where the object
// appeared in the source.  Keep their complete dictionary so extension ops
// cannot hide forbidden Types or Attributes in `<{...}>`.
//
// Do not de-duplicate registered properties by name or hasAttr() alone. Native
// (non-Attr) ODS properties need not be listed in the registered attribute
// metadata; a same-named discardable attribute can therefore make hasAttr(name)
// true while the property remains hidden.  The registered metadata plus exact
// Attribute equality is the stable storage-identity test for fields that
// getAttrs() really exposes.
Attribute getUncoveredOperationProperties(Operation *operation) {
  if (!operation->getPropertiesStorage())
    return {};
  Attribute properties = operation->getPropertiesAsAttribute();
  auto dictionary = dyn_cast_or_null<DictionaryAttr>(properties);
  if (!dictionary || !operation->isRegistered())
    return properties;
  auto registered = operation->getRegisteredInfo();
  if (!registered)
    return properties;
  llvm::ArrayRef<StringAttr> exposedNames = registered->getAttributeNames();
  llvm::SmallVector<NamedAttribute> uncovered;
  for (NamedAttribute entry : dictionary.getValue()) {
    if (!llvm::is_contained(exposedNames, entry.getName())) {
      uncovered.push_back(entry);
      continue;
    }
    Attribute visible = operation->getAttr(entry.getName());
    if (visible != entry.getValue())
      uncovered.push_back(entry);
  }
  if (uncovered.empty())
    return {};
  return DictionaryAttr::get(operation->getContext(), uncovered);
}

void diagnoseType(Type type, llvm::StringRef operationPath,
                  llvm::StringRef objectPath, Location location,
                  const ValidationPolicy &policy, ValidationReport &report) {
  llvm::StringRef dialect = getDialectNamespace(type);
  const DiagnosticTemplate *diagnosticTemplate = nullptr;
  if (policy.forbiddenDialects.count(dialect.str()))
    diagnosticTemplate = &policy.forbiddenType;
  else if (!isBuiltinObjectDialectOrAllowed(dialect, policy))
    diagnosticTemplate = &policy.unknownType;
  if (!diagnosticTemplate)
    return;

  std::string typeName = printIRObject(type);
  appendDiagnostic(policy, report,
                   makeTemplateDiagnostic(policy, *diagnosticTemplate, "type",
                                          typeName, operationPath,
                                          getSourceLocation(location), dialect,
                                          objectPath));
}

void diagnoseAttributeDialect(llvm::StringRef dialect,
                              llvm::StringRef attributeName,
                              llvm::StringRef operationPath,
                              llvm::StringRef objectPath, Location location,
                              const ValidationPolicy &policy,
                              ValidationReport &report) {
  const DiagnosticTemplate *diagnosticTemplate = nullptr;
  if (policy.forbiddenDialects.count(dialect.str()))
    diagnosticTemplate = &policy.forbiddenAttribute;
  else if (!isBuiltinObjectDialectOrAllowed(dialect, policy))
    diagnosticTemplate = &policy.unknownAttribute;
  if (!diagnosticTemplate)
    return;

  appendDiagnostic(
      policy, report,
      makeTemplateDiagnostic(policy, *diagnosticTemplate, "attribute",
                             attributeName, operationPath,
                             getSourceLocation(location), dialect, objectPath));
}

void diagnoseNamedAttribute(llvm::StringRef attributeName,
                            llvm::StringRef operationPath,
                            llvm::StringRef objectPath, Location location,
                            const ValidationPolicy &policy,
                            ValidationReport &report) {
  size_t separator = attributeName.find('.');
  if (separator == llvm::StringRef::npos)
    return;

  llvm::StringRef dialect = attributeName.take_front(separator);
  if (isValidNamedDialectNamespace(dialect)) {
    diagnoseAttributeDialect(dialect, attributeName, operationPath, objectPath,
                             location, policy, report);
    return;
  }

  appendDiagnostic(policy, report,
                   makeTemplateDiagnostic(
                       policy, policy.unknownAttribute, "attribute",
                       attributeName, operationPath,
                       getSourceLocation(location), "<empty>", objectPath));
}

void diagnoseAttribute(Attribute attribute, llvm::StringRef operationPath,
                       llvm::StringRef objectPath, Location location,
                       const ValidationPolicy &policy,
                       ValidationReport &report) {
  std::string attributeName = printIRObject(attribute);
  diagnoseAttributeDialect(getDialectNamespace(attribute), attributeName,
                           operationPath, objectPath, location, policy, report);
}

void visitType(Type type, llvm::StringRef operationPath,
               llvm::StringRef objectPath, Location location, ModuleOp module,
               const ValidationPolicy &policy, ValidationReport &report,
               ObjectTraversalState &state) {
  if (report.resourceLimitReported)
    return;
  if (report.objectVisitCount >= kMaxAnchorIRObjectVisits) {
    appendResourceLimitDiagnostic(
        policy, report, "type/attribute traversal count", "builtin.module",
        getSourceLocation(location));
    return;
  }
  ++report.objectVisitCount;
  if (state.depth >= kMaxAnchorIRStructuralDepth) {
    appendResourceLimitDiagnostic(policy, report, "nested object depth",
                                  operationPath, getSourceLocation(location));
    return;
  }
  ++state.depth;
  auto restoreDepth = llvm::make_scope_exit([&]() { --state.depth; });

  const void *identity = type.getAsOpaquePointer();
  if (!state.activeTypes.insert(identity).second)
    return;
  auto removeActive =
      llvm::make_scope_exit([&]() { state.activeTypes.erase(identity); });
  // Printing and semantic validation may recursively inspect the complete
  // object.  Defer both until the bounded child walk proves that the object is
  // within the structural limit.
  auto validateCurrent = llvm::make_scope_exit([&]() {
    if (report.resourceLimitReported)
      return;
    diagnoseType(type, operationPath, objectPath, location, policy, report);
    if (report.resourceLimitReported)
      return;
    validateTritonGPUShapedType(type, operationPath, objectPath, location,
                                module, policy, report);
  });

  if (auto function = dyn_cast<FunctionType>(type)) {
    for (auto entry : llvm::enumerate(function.getInputs())) {
      visitType(entry.value(), operationPath,
                (llvm::Twine(objectPath) + ".input[" +
                 llvm::Twine(entry.index()) + "]")
                    .str(),
                location, module, policy, report, state);
      if (report.resourceLimitReported)
        return;
    }
    for (auto entry : llvm::enumerate(function.getResults())) {
      visitType(entry.value(), operationPath,
                (llvm::Twine(objectPath) + ".result[" +
                 llvm::Twine(entry.index()) + "]")
                    .str(),
                location, module, policy, report, state);
      if (report.resourceLimitReported)
        return;
    }
    return;
  }

  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    visitType(tensor.getElementType(), operationPath,
              (llvm::Twine(objectPath) + ".element_type").str(), location,
              module, policy, report, state);
    if (report.resourceLimitReported)
      return;
    if (Attribute encoding = tensor.getEncoding())
      visitAttribute(encoding, operationPath,
                     (llvm::Twine(objectPath) + ".encoding").str(), location,
                     module, policy, report, state);
    return;
  }

  if (auto tensor = dyn_cast<UnrankedTensorType>(type)) {
    visitType(tensor.getElementType(), operationPath,
              (llvm::Twine(objectPath) + ".element_type").str(), location,
              module, policy, report, state);
    return;
  }

  if (auto memref = dyn_cast<MemRefType>(type)) {
    visitType(memref.getElementType(), operationPath,
              (llvm::Twine(objectPath) + ".element_type").str(), location,
              module, policy, report, state);
    if (report.resourceLimitReported)
      return;
    if (Attribute layout = memref.getLayout())
      visitAttribute(layout, operationPath,
                     (llvm::Twine(objectPath) + ".layout").str(), location,
                     module, policy, report, state);
    if (report.resourceLimitReported)
      return;
    if (Attribute memorySpace = memref.getMemorySpace())
      visitAttribute(memorySpace, operationPath,
                     (llvm::Twine(objectPath) + ".memory_space").str(),
                     location, module, policy, report, state);
    return;
  }

  if (auto memref = dyn_cast<UnrankedMemRefType>(type)) {
    visitType(memref.getElementType(), operationPath,
              (llvm::Twine(objectPath) + ".element_type").str(), location,
              module, policy, report, state);
    if (report.resourceLimitReported)
      return;
    if (Attribute memorySpace = memref.getMemorySpace())
      visitAttribute(memorySpace, operationPath,
                     (llvm::Twine(objectPath) + ".memory_space").str(),
                     location, module, policy, report, state);
    return;
  }

  if (auto tuple = dyn_cast<TupleType>(type)) {
    for (auto entry : llvm::enumerate(tuple.getTypes())) {
      visitType(entry.value(), operationPath,
                (llvm::Twine(objectPath) + ".element[" +
                 llvm::Twine(entry.index()) + "]")
                    .str(),
                location, module, policy, report, state);
      if (report.resourceLimitReported)
        return;
    }
    return;
  }

  if (auto memdesc = dyn_cast<triton::MemDescType>(type)) {
    visitType(memdesc.getElementType(), operationPath,
              (llvm::Twine(objectPath) + ".element_type").str(), location,
              module, policy, report, state);
    if (report.resourceLimitReported)
      return;
    if (Attribute encoding = memdesc.getEncoding())
      visitAttribute(encoding, operationPath,
                     (llvm::Twine(objectPath) + ".encoding").str(), location,
                     module, policy, report, state);
    return;
  }

  if (auto shaped = dyn_cast<ShapedType>(type)) {
    visitType(shaped.getElementType(), operationPath,
              (llvm::Twine(objectPath) + ".element_type").str(), location,
              module, policy, report, state);
    return;
  }

  unsigned attributeIndex = 0;
  unsigned typeIndex = 0;
  type.walkImmediateSubElements(
      [&](Attribute child) {
        if (report.resourceLimitReported)
          return;
        visitAttribute(child, operationPath,
                       (llvm::Twine(objectPath) + ".attribute[" +
                        llvm::Twine(attributeIndex++) + "]")
                           .str(),
                       location, module, policy, report, state);
      },
      [&](Type child) {
        if (report.resourceLimitReported)
          return;
        visitType(child, operationPath,
                  (llvm::Twine(objectPath) + ".type[" +
                   llvm::Twine(typeIndex++) + "]")
                      .str(),
                  location, module, policy, report, state);
      });
}

void visitAttribute(Attribute attribute, llvm::StringRef operationPath,
                    llvm::StringRef objectPath, Location location,
                    ModuleOp module, const ValidationPolicy &policy,
                    ValidationReport &report, ObjectTraversalState &state) {
  if (report.resourceLimitReported)
    return;
  if (report.objectVisitCount >= kMaxAnchorIRObjectVisits) {
    appendResourceLimitDiagnostic(
        policy, report, "type/attribute traversal count", "builtin.module",
        getSourceLocation(location));
    return;
  }
  ++report.objectVisitCount;
  if (state.depth >= kMaxAnchorIRStructuralDepth) {
    appendResourceLimitDiagnostic(policy, report, "nested object depth",
                                  operationPath, getSourceLocation(location));
    return;
  }
  ++state.depth;
  auto restoreDepth = llvm::make_scope_exit([&]() { --state.depth; });

  const void *identity = attribute.getAsOpaquePointer();
  if (!state.activeAttributes.insert(identity).second)
    return;
  auto removeActive =
      llvm::make_scope_exit([&]() { state.activeAttributes.erase(identity); });
  // Avoid printing a recursively nested attribute before its bounded child
  // traversal has established that doing so is safe.
  auto diagnoseCurrent = llvm::make_scope_exit([&]() {
    if (!report.resourceLimitReported)
      diagnoseAttribute(attribute, operationPath, objectPath, location, policy,
                        report);
  });

  // Typed attributes carry a Type that is not required to be exposed through
  // walkImmediateSubElements().  DenseElementsAttr is the important example:
  // a forbidden tensor encoding or element type can otherwise hide entirely
  // inside the attribute's shaped type.
  Type typedAttributeType;
  if (auto typedAttribute = dyn_cast<TypedAttr>(attribute)) {
    typedAttributeType = typedAttribute.getType();
    visitType(typedAttributeType, operationPath,
              (llvm::Twine(objectPath) + ".type").str(), location, module,
              policy, report, state);
    if (report.resourceLimitReported)
      return;
  }

  if (auto array = dyn_cast<ArrayAttr>(attribute)) {
    for (auto entry : llvm::enumerate(array.getValue())) {
      visitAttribute(entry.value(), operationPath,
                     (llvm::Twine(objectPath) + ".element[" +
                      llvm::Twine(entry.index()) + "]")
                         .str(),
                     location, module, policy, report, state);
      if (report.resourceLimitReported)
        return;
    }
    return;
  }

  if (auto dictionary = dyn_cast<DictionaryAttr>(attribute)) {
    for (NamedAttribute entry : dictionary.getValue()) {
      std::string entryPath =
          (llvm::Twine(objectPath) + ".entry[" +
           escapePathComponent(entry.getName().strref()) + "]")
              .str();
      diagnoseNamedAttribute(entry.getName().strref(), operationPath, entryPath,
                             location, policy, report);
      if (report.resourceLimitReported)
        return;
      visitAttribute(entry.getValue(), operationPath, entryPath, location,
                     module, policy, report, state);
      if (report.resourceLimitReported)
        return;
    }
    return;
  }

  if (auto typeAttribute = dyn_cast<TypeAttr>(attribute)) {
    visitType(typeAttribute.getValue(), operationPath,
              (llvm::Twine(objectPath) + ".value").str(), location, module,
              policy, report, state);
    return;
  }

  unsigned attributeIndex = 0;
  unsigned typeIndex = 0;
  attribute.walkImmediateSubElements(
      [&](Attribute child) {
        if (report.resourceLimitReported)
          return;
        visitAttribute(child, operationPath,
                       (llvm::Twine(objectPath) + ".attribute[" +
                        llvm::Twine(attributeIndex++) + "]")
                           .str(),
                       location, module, policy, report, state);
      },
      [&](Type child) {
        if (report.resourceLimitReported)
          return;
        // Avoid diagnosing the same direct TypedAttr type twice when an
        // Attribute implementation also exposes it as an immediate subelement.
        if (child == typedAttributeType)
          return;
        visitType(child, operationPath,
                  (llvm::Twine(objectPath) + ".type[" +
                   llvm::Twine(typeIndex++) + "]")
                      .str(),
                  location, module, policy, report, state);
      });
}

void visitOperationObjects(Operation *operation, llvm::StringRef operationPath,
                           ModuleOp module, const ValidationPolicy &policy,
                           ValidationReport &report) {
  if (report.resourceLimitReported)
    return;
  ObjectTraversalState state;
  Location location = operation->getLoc();

  for (auto entry : llvm::enumerate(operation->getOperands())) {
    visitType(
        entry.value().getType(), operationPath,
        (llvm::Twine("operand[") + llvm::Twine(entry.index()) + "].type").str(),
        location, module, policy, report, state);
    if (report.resourceLimitReported)
      return;
  }
  for (auto entry : llvm::enumerate(operation->getResults())) {
    visitType(
        entry.value().getType(), operationPath,
        (llvm::Twine("result[") + llvm::Twine(entry.index()) + "].type").str(),
        location, module, policy, report, state);
    if (report.resourceLimitReported)
      return;
  }
  for (NamedAttribute attribute : operation->getAttrs()) {
    llvm::StringRef attributeName = attribute.getName().strref();
    std::string attributePath =
        (llvm::Twine("attribute[") + escapePathComponent(attributeName) + "]")
            .str();
    diagnoseNamedAttribute(attributeName, operationPath, attributePath,
                           location, policy, report);
    if (report.resourceLimitReported)
      return;
    visitAttribute(attribute.getValue(), operationPath, attributePath, location,
                   module, policy, report, state);
    if (report.resourceLimitReported)
      return;
  }

  // Generic and registered operations may keep semantic properties in a
  // separate opaque slot.  The helper removes entries already exposed through
  // getAttrs() while retaining genuinely hidden properties.
  if (Attribute properties = getUncoveredOperationProperties(operation))
    visitAttribute(properties, operationPath, "properties", location, module,
                   policy, report, state);
}

bool isUnrankedShapedValue(Type type) {
  return isa<UnrankedTensorType, UnrankedMemRefType>(type);
}

bool isTritonGPUOperation(Operation *operation) {
  llvm::StringRef dialect = operation->getName().getDialectNamespace();
  return dialect == "tt" || dialect == "triton_gpu";
}

void validateLinalgSemantics(Operation *operation,
                             llvm::StringRef operationPath,
                             const ValidationPolicy &policy,
                             ValidationReport &report) {
  llvm::StringRef operationName = operation->getName().getStringRef();
  llvm::StringRef dialect = operation->getName().getDialectNamespace();
  if (hasInvariant(policy, "linalg.no_unrealized_conversion_cast") &&
      operationName == "builtin.unrealized_conversion_cast")
    diagnoseSemantic(policy, "linalg.no_unrealized_conversion_cast",
                     "operation", operationName, operationPath,
                     operation->getLoc(), report);
  if (report.resourceLimitReported)
    return;

  if (dialect != "linalg" && dialect != "linalg_ext")
    return;
  if (hasInvariant(policy, "linalg.ranked_shaped_values")) {
    for (auto entry : llvm::enumerate(operation->getOperands())) {
      if (isUnrankedShapedValue(entry.value().getType())) {
        std::string object =
            (llvm::Twine("operand[") + llvm::Twine(entry.index()) + "]").str();
        diagnoseSemantic(policy, "linalg.ranked_shaped_values", "type", object,
                         operationPath, operation->getLoc(), report,
                         (object + ".type"), operationName);
        if (report.resourceLimitReported)
          return;
      }
    }
    for (auto entry : llvm::enumerate(operation->getResults())) {
      if (isUnrankedShapedValue(entry.value().getType())) {
        std::string object =
            (llvm::Twine("result[") + llvm::Twine(entry.index()) + "]").str();
        diagnoseSemantic(policy, "linalg.ranked_shaped_values", "type", object,
                         operationPath, operation->getLoc(), report,
                         (object + ".type"), operationName);
        if (report.resourceLimitReported)
          return;
      }
    }
  }

  if (!hasInvariant(policy, "linalg.generic_region_contract") ||
      operationName != "linalg.generic")
    return;
  bool valid = operation->getNumRegions() == 1;
  if (valid) {
    Region &region = operation->getRegion(0);
    valid = llvm::hasSingleElement(region);
    if (valid) {
      Block &block = region.front();
      // `getTerminator()` asserts when a malformed, unverified block is
      // empty or ends in a non-terminator.  AnchorIR validation deliberately
      // runs before MLIR's verifier, so inspect the block safely first and
      // turn malformed structure into the normal semantic diagnostic.
      Operation *terminator =
          !block.empty() && block.back().mightHaveTrait<OpTrait::IsTerminator>()
              ? &block.back()
              : nullptr;
      auto generic = dyn_cast<linalg::GenericOp>(operation);
      valid = terminator && generic &&
              terminator->getName().getStringRef() == "linalg.yield" &&
              block.getNumArguments() == operation->getNumOperands() &&
              terminator->getNumOperands() == generic.getNumDpsInits();
    }
  }
  if (!valid)
    diagnoseSemantic(policy, "linalg.generic_region_contract", "operation",
                     operationName, operationPath, operation->getLoc(), report);
}

void validateTritonGPUSemantics(Operation *operation,
                                llvm::StringRef operationPath,
                                const ValidationPolicy &policy,
                                ValidationReport &report) {
  if (!isTritonGPUOperation(operation))
    return;
  llvm::StringRef name = operation->getName().getStringRef();

  if (hasInvariant(policy, "gpu.operation_contract")) {
    bool operationSafe = true;
    llvm::StringRef objectPath;
    if (name == "tt.elementwise_inline_asm") {
      objectPath = "attribute[packed_element]";
      auto packed = operation->getAttrOfType<IntegerAttr>("packed_element");
      operationSafe = packed && !packed.getValue().isZero() &&
                      !packed.getValue().isNegative();
    } else if (name == "tt.reshape" && operation->getNumOperands() >= 1) {
      Attribute sourceEncoding =
          getTritonGPUShapedEncoding(operation->getOperand(0).getType());
      auto allowReorder = operation->getAttrOfType<BoolAttr>("allow_reorder");
      if (sourceEncoding && allowReorder && !allowReorder.getValue()) {
        objectPath = "operand[0].type.encoding";
        operationSafe = isa<triton::DialectInferLayoutInterface>(
            &sourceEncoding.getDialect());
      }
    } else if (name == "tt.reduce" || name == "tt.scan") {
      objectPath = "attribute[axis]";
      auto axisAttribute = operation->getAttrOfType<IntegerAttr>("axis");
      std::optional<unsigned> axis = getNonNegativeUnsigned(axisAttribute);
      operationSafe = axis && operation->getNumOperands() != 0 &&
                      operation->getNumOperands() == operation->getNumResults();
      std::optional<llvm::ArrayRef<int64_t>> commonShape;
      for (unsigned index = 0;
           operationSafe && index < operation->getNumOperands(); ++index) {
        auto source =
            dyn_cast<RankedTensorType>(operation->getOperand(index).getType());
        if (!source || *axis >= source.getRank()) {
          operationSafe = false;
          break;
        }
        if (commonShape && *commonShape != source.getShape()) {
          operationSafe = false;
          break;
        }
        commonShape = source.getShape();

        Type result = operation->getResult(index).getType();
        if (name == "tt.scan") {
          operationSafe = result == source;
          continue;
        }
        llvm::SmallVector<int64_t> expectedShape(source.getShape());
        expectedShape.erase(expectedShape.begin() + *axis);
        if (expectedShape.empty()) {
          operationSafe = result == source.getElementType();
          continue;
        }
        auto resultTensor = dyn_cast<RankedTensorType>(result);
        operationSafe =
            resultTensor &&
            resultTensor.getShape() == llvm::ArrayRef<int64_t>(expectedShape) &&
            resultTensor.getElementType() == source.getElementType();
      }
    } else if (name == "tt.expand_dims") {
      objectPath = "attribute[axis]";
      auto axisAttribute = operation->getAttrOfType<IntegerAttr>("axis");
      auto source =
          operation->getNumOperands() == 1
              ? dyn_cast<RankedTensorType>(operation->getOperand(0).getType())
              : RankedTensorType();
      auto result =
          operation->getNumResults() == 1
              ? dyn_cast<RankedTensorType>(operation->getResult(0).getType())
              : RankedTensorType();
      std::optional<unsigned> axis = getNonNegativeUnsigned(axisAttribute);
      operationSafe = source && result && axis &&
                      *axis <= static_cast<unsigned>(source.getRank());
      if (operationSafe) {
        llvm::SmallVector<int64_t> expectedShape(source.getShape());
        expectedShape.insert(expectedShape.begin() + *axis, 1);
        operationSafe =
            result.getShape() == llvm::ArrayRef<int64_t>(expectedShape) &&
            result.getElementType() == source.getElementType();
      }
      if (operationSafe) {
        Attribute sourceEncoding = source.getEncoding();
        Attribute resultEncoding = result.getEncoding();
        if (sourceEncoding &&
            getDialectNamespace(sourceEncoding) == "triton_gpu")
          operationSafe =
              !isa<triton::gpu::SharedEncodingAttr>(sourceEncoding) &&
              !isa<triton::gpu::SharedEncodingAttr>(resultEncoding) &&
              hasSafeCoreTritonGPUEncoding(sourceEncoding, source.getShape()) &&
              hasSafeCoreTritonGPUEncoding(resultEncoding, result.getShape());
        auto *interface = operationSafe && sourceEncoding
                              ? dyn_cast<triton::DialectInferLayoutInterface>(
                                    &sourceEncoding.getDialect())
                              : nullptr;
        Attribute inferred;
        operationSafe =
            sourceEncoding && resultEncoding && interface &&
            succeeded(interface->inferExpandDimsOpEncoding(
                sourceEncoding, *axis, inferred, operation->getLoc())) &&
            inferred == resultEncoding;
      }
    } else if (name == "tt.broadcast") {
      objectPath = "result[0].type";
      auto source =
          operation->getNumOperands() == 1
              ? dyn_cast<RankedTensorType>(operation->getOperand(0).getType())
              : RankedTensorType();
      auto result =
          operation->getNumResults() == 1
              ? dyn_cast<RankedTensorType>(operation->getResult(0).getType())
              : RankedTensorType();
      operationSafe = source && result &&
                      source.getRank() == result.getRank() &&
                      source.getElementType() == result.getElementType();
      for (int64_t index = 0; operationSafe && index < source.getRank();
           ++index)
        operationSafe = source.getDimSize(index) == result.getDimSize(index) ||
                        source.getDimSize(index) == 1;
    } else if (name == "tt.cat") {
      objectPath = "result[0].type";
      auto lhs =
          operation->getNumOperands() == 2
              ? dyn_cast<RankedTensorType>(operation->getOperand(0).getType())
              : RankedTensorType();
      auto rhs =
          operation->getNumOperands() == 2
              ? dyn_cast<RankedTensorType>(operation->getOperand(1).getType())
              : RankedTensorType();
      auto result =
          operation->getNumResults() == 1
              ? dyn_cast<RankedTensorType>(operation->getResult(0).getType())
              : RankedTensorType();
      operationSafe = lhs && rhs && result && lhs == rhs &&
                      lhs.getRank() == 1 && result.getRank() == 1 &&
                      result.getElementType() == lhs.getElementType();
      if (operationSafe) {
        int64_t leading = lhs.getDimSize(0);
        operationSafe =
            leading <= std::numeric_limits<int64_t>::max() - leading &&
            result.getDimSize(0) == leading + leading;
      }
    } else if (name == "tt.histogram") {
      objectPath = "result[0].type";
      auto source =
          operation->getNumOperands() == 1
              ? dyn_cast<RankedTensorType>(operation->getOperand(0).getType())
              : RankedTensorType();
      auto result =
          operation->getNumResults() == 1
              ? dyn_cast<RankedTensorType>(operation->getResult(0).getType())
              : RankedTensorType();
      operationSafe = source && result && source.getRank() == 1 &&
                      result.getRank() == 1 &&
                      source.getElementType().isIntOrIndex() &&
                      result.getElementType().isSignlessInteger(32);
    } else if (name == "tt.make_tensor_ptr") {
      objectPath = "attribute[order]";
      auto resultPointer =
          operation->getNumResults() == 1
              ? dyn_cast<triton::PointerType>(operation->getResult(0).getType())
              : triton::PointerType();
      auto pointee =
          resultPointer
              ? dyn_cast<RankedTensorType>(resultPointer.getPointeeType())
              : RankedTensorType();
      auto order = operation->getAttrOfType<DenseI32ArrayAttr>("order");
      unsigned rank = pointee ? static_cast<unsigned>(pointee.getRank()) : 0;
      operationSafe = pointee && rank != 0 &&
                      operation->getNumOperands() == 1 + 3 * rank && order &&
                      isPermutationOfRank(order.asArrayRef(), rank);
    } else if (name == "tt.advance") {
      objectPath = "operand[1]";
      std::optional<unsigned> rank =
          operation->getNumOperands() != 0
              ? getTensorPointerRank(operation->getOperand(0).getType())
              : std::nullopt;
      operationSafe = rank && operation->getNumResults() == 1 &&
                      operation->getNumOperands() == 1 + *rank &&
                      operation->getResult(0).getType() ==
                          operation->getOperand(0).getType();
    } else if (name == "tt.load" || name == "tt.store") {
      objectPath = "attribute[boundaryCheck]";
      bool blockPointer =
          operation->getNumOperands() != 0 &&
          getTensorPointerRank(operation->getOperand(0).getType()).has_value();
      unsigned expectedBlockPointerOperands = name == "tt.load" ? 1 : 2;
      operationSafe =
          hasValidBoundaryCheck(operation, /*pointerOperandIndex=*/0) &&
          (!blockPointer ||
           operation->getNumOperands() == expectedBlockPointerOperands);
      if (operationSafe && blockPointer && name == "tt.load") {
        auto load = dyn_cast<triton::LoadOp>(operation);
        auto pointer =
            dyn_cast<triton::PointerType>(operation->getOperand(0).getType());
        auto pointee =
            pointer ? dyn_cast<RankedTensorType>(pointer.getPointeeType())
                    : RankedTensorType();
        std::optional<triton::PaddingOption> padding =
            load ? load.getPadding() : std::nullopt;
        operationSafe = !padding ||
                        *padding != triton::PaddingOption::PAD_NAN ||
                        (pointee && !pointee.getElementType().isIntOrIndex());
      }
    } else if (name == "triton_gpu.async_wait") {
      objectPath = "attribute[num]";
      operationSafe =
          getNonNegativeUnsigned(operation->getAttrOfType<IntegerAttr>("num"))
              .has_value();
    } else if (name == "tt.dot") {
      objectPath = "attribute[maxNumImpreciseAcc]";
      operationSafe =
          getNonNegativeUnsigned(
              operation->getAttrOfType<IntegerAttr>("maxNumImpreciseAcc"))
              .has_value();
    }
    if (!operationSafe)
      diagnoseSemantic(policy, "gpu.operation_contract", "operation", name,
                       operationPath, operation->getLoc(), report, objectPath,
                       name);
  }

  if (!hasInvariant(policy, "gpu.dot_encoding_contract") || name != "tt.dot")
    return;
  bool valid =
      operation->getNumOperands() == 3 && operation->getNumResults() == 1;
  Attribute resultEncoding;
  Attribute accumulatorEncoding;
  Attribute aEncoding;
  Attribute bEncoding;
  std::optional<llvm::ArrayRef<int64_t>> aShape;
  std::optional<llvm::ArrayRef<int64_t>> bShape;
  std::optional<llvm::ArrayRef<int64_t>> accumulatorShape;
  std::optional<llvm::ArrayRef<int64_t>> resultShape;
  if (valid) {
    auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
    resultEncoding = result ? result.getEncoding() : Attribute();
    auto accumulator =
        dyn_cast<RankedTensorType>(operation->getOperand(2).getType());
    accumulatorEncoding = accumulator ? accumulator.getEncoding() : Attribute();
    aEncoding = getTritonGPUShapedEncoding(operation->getOperand(0).getType());
    bEncoding = getTritonGPUShapedEncoding(operation->getOperand(1).getType());
    aShape = getTritonGPUShapedShape(operation->getOperand(0).getType());
    bShape = getTritonGPUShapedShape(operation->getOperand(1).getType());
    accumulatorShape =
        getTritonGPUShapedShape(operation->getOperand(2).getType());
    resultShape = getTritonGPUShapedShape(operation->getResult(0).getType());
    valid = resultEncoding && accumulatorEncoding &&
            resultEncoding == accumulatorEncoding &&
            !isa<triton::gpu::DotOperandEncodingAttr>(resultEncoding) &&
            aEncoding && bEncoding && aShape && bShape && accumulatorShape &&
            resultShape;
  }
  if (valid) {
    auto encodingSafe = [](Attribute encoding, llvm::ArrayRef<int64_t> shape) {
      return getDialectNamespace(encoding) != "triton_gpu" ||
             hasSafeCoreTritonGPUEncoding(encoding, shape);
    };
    valid = encodingSafe(aEncoding, *aShape) &&
            encodingSafe(bEncoding, *bShape) &&
            encodingSafe(accumulatorEncoding, *accumulatorShape) &&
            encodingSafe(resultEncoding, *resultShape);
  }
  if (valid) {
    Dialect &dialect = aEncoding.getDialect();
    auto *interface = dyn_cast<triton::DialectInferLayoutInterface>(&dialect);
    valid = interface &&
            succeeded(interface->inferDotOpEncoding(
                aEncoding, /*opIdx=*/0, resultEncoding, std::nullopt)) &&
            succeeded(interface->inferDotOpEncoding(
                bEncoding, /*opIdx=*/1, resultEncoding, std::nullopt));
  }
  if (valid) {
    valid = (aShape->size() == 2 || aShape->size() == 3) &&
            aShape->size() == bShape->size() &&
            aShape->size() == accumulatorShape->size() &&
            aShape->size() == resultShape->size() &&
            *accumulatorShape == *resultShape;
    if (valid) {
      size_t rank = aShape->size();
      for (size_t index = 0; index + 2 < rank; ++index)
        valid = valid && (*aShape)[index] == (*bShape)[index] &&
                (*aShape)[index] == (*resultShape)[index];
      valid = valid && (*aShape)[rank - 2] == (*resultShape)[rank - 2] &&
              (*aShape)[rank - 1] == (*bShape)[rank - 2] &&
              (*bShape)[rank - 1] == (*resultShape)[rank - 1];
    }
  }
#if defined(TRITON_ANCHOR_HAS_FANT_WMMA)
  if (valid) {
    auto fantParent =
        [](Attribute encoding) -> triton::gpu::FANTWmmaEncodingAttr {
      if (auto dot = dyn_cast<triton::gpu::DotOperandEncodingAttr>(encoding))
        return dyn_cast<triton::gpu::FANTWmmaEncodingAttr>(dot.getParent());
      return dyn_cast<triton::gpu::FANTWmmaEncodingAttr>(encoding);
    };
    bool usesFANT = fantParent(aEncoding) || fantParent(bEncoding) ||
                    fantParent(resultEncoding);
    if (usesFANT) {
      auto dot = dyn_cast<triton::DotOp>(operation);
      auto aType =
          dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
      auto bType =
          dyn_cast<RankedTensorType>(operation->getOperand(1).getType());
      auto cType =
          dyn_cast<RankedTensorType>(operation->getOperand(2).getType());
      auto dType =
          dyn_cast<RankedTensorType>(operation->getResult(0).getType());
      valid = dot && aType && bType && cType && dType &&
              fantParent(aEncoding) && fantParent(bEncoding) &&
              fantParent(resultEncoding) && fantParent(accumulatorEncoding);
      if (valid) {
        Type aElement = aType.getElementType();
        Type bElement = bType.getElementType();
        Type cElement = cType.getElementType();
        Type dElement = dType.getElementType();
        int64_t m = (*resultShape)[resultShape->size() - 2];
        int64_t n = (*resultShape)[resultShape->size() - 1];
        int64_t k = (*aShape)[aShape->size() - 1];
        bool shapeSupported = m % 16 == 0 && n % 16 == 0;
        bool typeSupported = false;
        if (aElement.isF16() || aElement.isBF16()) {
          typeSupported = bElement == aElement && cElement == dElement &&
                          (cElement == aElement || cElement.isF32()) &&
                          k % 32 == 0;
        } else if (aElement.isF32()) {
          typeSupported =
              bElement.isF32() && cElement.isF32() && dElement.isF32() &&
              dot.getInputPrecision() == triton::InputPrecision::TF32 &&
              k % 16 == 0;
        } else if (aElement.isSignlessInteger(8)) {
          typeSupported = bElement.isSignlessInteger(8) &&
                          cElement.isSignlessInteger(32) &&
                          dElement.isSignlessInteger(32) && k % 64 == 0;
        }
        valid = shapeSupported && typeSupported;
      }
    }
  }
#endif
  if (!valid)
    diagnoseSemantic(policy, "gpu.dot_encoding_contract", "operation", name,
                     operationPath, operation->getLoc(), report);
}

std::optional<int> getPositiveIntegerAttribute(ModuleOp module,
                                               llvm::StringRef name) {
  auto attribute = module->getAttrOfType<IntegerAttr>(name);
  if (!attribute)
    return std::nullopt;

  const llvm::APInt &value = attribute.getValue();
  if (value.isZero() || value.isNegative() ||
      value.getActiveBits() >
          static_cast<unsigned>(std::numeric_limits<int>::digits))
    return std::nullopt;
  return static_cast<int>(value.getZExtValue());
}

bool containsTritonGPUObject(ModuleOp module) {
  bool found = false;
  llvm::SmallPtrSet<const void *, 16> activeTypes;
  llvm::SmallPtrSet<const void *, 16> activeAttributes;
  std::function<void(Type)> containsType;
  std::function<void(Attribute)> containsAttribute;

  containsType = [&](Type object) {
    if (found || !object)
      return;
    if (getDialectNamespace(object) == "triton_gpu") {
      found = true;
      return;
    }
    const void *identity = object.getAsOpaquePointer();
    if (!activeTypes.insert(identity).second)
      return;
    auto removeActive =
        llvm::make_scope_exit([&]() { activeTypes.erase(identity); });
    object.walkImmediateSubElements(containsAttribute, containsType);
  };

  containsAttribute = [&](Attribute object) {
    if (found || !object)
      return;
    if (getDialectNamespace(object) == "triton_gpu") {
      found = true;
      return;
    }
    const void *identity = object.getAsOpaquePointer();
    if (!activeAttributes.insert(identity).second)
      return;
    auto removeActive =
        llvm::make_scope_exit([&]() { activeAttributes.erase(identity); });

    Type typedAttributeType;
    if (auto typedAttribute = dyn_cast<TypedAttr>(object)) {
      typedAttributeType = typedAttribute.getType();
      containsType(typedAttributeType);
    }
    object.walkImmediateSubElements(containsAttribute, [&](Type child) {
      if (child != typedAttributeType)
        containsType(child);
    });
  };

  module.walk([&](Operation *operation) {
    if (found)
      return;
    if (isTritonGPUOperation(operation)) {
      found = true;
      return;
    }
    for (Type type : operation->getOperandTypes())
      containsType(type);
    for (Type type : operation->getResultTypes())
      containsType(type);
    for (NamedAttribute attribute : operation->getAttrs())
      containsAttribute(attribute.getValue());
    if (Attribute properties = getUncoveredOperationProperties(operation))
      containsAttribute(properties);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          containsType(argument.getType());
  });
  return found;
}

bool validateGPUConfiguration(ModuleOp module, const ValidationPolicy &policy,
                              ValidationReport &report) {
  bool hasConfigurationAttribute =
      module->hasAttr("triton_gpu.num-warps") ||
      module->hasAttr("triton_gpu.threads-per-warp") ||
      module->hasAttr("triton_gpu.num-ctas");
  if (!hasInvariant(policy, "gpu.module_configuration") ||
      (!containsTritonGPUObject(module) && !hasConfigurationAttribute))
    return true;
  auto numWarps = getPositiveIntegerAttribute(module, "triton_gpu.num-warps");
  auto threadsPerWarp =
      getPositiveIntegerAttribute(module, "triton_gpu.threads-per-warp");
  auto numCTAs = getPositiveIntegerAttribute(module, "triton_gpu.num-ctas");
  bool valid = true;
  auto diagnoseInvalid = [&](llvm::StringRef name,
                             const std::optional<int> &value) {
    if (value)
      return;
    valid = false;
    diagnoseSemantic(policy, "gpu.module_configuration", "attribute", name,
                     "builtin.module", module->getLoc(), report,
                     (llvm::Twine("attribute[") + name + "]").str());
  };
  diagnoseInvalid("triton_gpu.num-warps", numWarps);
  diagnoseInvalid("triton_gpu.threads-per-warp", threadsPerWarp);
  diagnoseInvalid("triton_gpu.num-ctas", numCTAs);
  return valid;
}

bool isTritonDotVerifierSafe(ModuleOp module) {
  bool safe = true;
  module.walk([&](Operation *operation) {
    if (!safe || operation->getName().getStringRef() != "tt.dot" ||
        operation->getNumOperands() < 2)
      return;

    Attribute aEncoding =
        getTritonGPUShapedEncoding(operation->getOperand(0).getType());
    Attribute bEncoding =
        getTritonGPUShapedEncoding(operation->getOperand(1).getType());
    // DotOp::verify handles the both-missing and one-missing cases without
    // consulting a dialect interface.  When A has an encoding, however, the
    // pinned verifier uses llvm::cast<DialectInferLayoutInterface> and aborts
    // for builtin or opaque encodings instead of returning failure.
    if (!aEncoding || !bEncoding)
      return;
    Dialect &dialect = aEncoding.getDialect();
    if (!isa<triton::DialectInferLayoutInterface>(&dialect)) {
      safe = false;
      return;
    }
    Dialect &bDialect = bEncoding.getDialect();
    if (!isa<triton::DialectInferLayoutInterface>(&bDialect)) {
      safe = false;
      return;
    }

    // The pinned TritonGPU verifier dereferences A's DotOperand wrapper when
    // exactly B is a DotOperand.  A shared/other layout plus a DotOperand B is
    // invalid, but must become AIR-GPU-013 instead of a native segfault.
    if (getDialectNamespace(aEncoding) == "triton_gpu" &&
        !isa<triton::gpu::DotOperandEncodingAttr>(aEncoding) &&
        isa<triton::gpu::DotOperandEncodingAttr>(bEncoding))
      safe = false;
  });
  return safe;
}

bool isTritonReshapeVerifierSafe(ModuleOp module) {
  bool safe = true;
  module.walk([&](Operation *operation) {
    if (!safe || operation->getName().getStringRef() != "tt.reshape")
      return;
    // ReshapeOp's generated accessors and verifier assume its ODS
    // one-operand/one-result contract. Generic IR can violate that contract
    // before verification, so reject it before any accessor indexes a range.
    safe = operation->getNumOperands() == 1 &&
           operation->getNumResults() == 1;
  });
  return safe;
}

enum class StructuralLimitKind { OperationDepth, OperationCount };

struct StructuralLimit {
  StructuralLimitKind kind;
};

std::optional<StructuralLimit> findOperationStructuralLimit(ModuleOp module) {
  // Do not use Operation::walk here.  This preflight must itself remain safe
  // for the deeply nested module that it is intended to reject.
  std::vector<std::pair<Operation *, unsigned>> pending;
  pending.emplace_back(module.getOperation(), 0);
  size_t operationCount = 0;
  while (!pending.empty()) {
    auto [operation, depth] = pending.back();
    pending.pop_back();
    for (Region &region : operation->getRegions()) {
      for (Block &block : region) {
        for (Operation &child : block) {
          unsigned childDepth = depth + 1;
          if (childDepth > kMaxAnchorIRStructuralDepth)
            return StructuralLimit{StructuralLimitKind::OperationDepth};
          if (operationCount == kMaxAnchorIROperations)
            return StructuralLimit{StructuralLimitKind::OperationCount};
          ++operationCount;
          pending.emplace_back(&child, childDepth);
        }
      }
    }
  }
  return std::nullopt;
}

std::string getOperationSegment(Operation *operation, unsigned ordinal) {
  std::string segment = operation->getName().getStringRef().str();
  if (auto symbol = operation->getAttrOfType<StringAttr>(
          SymbolTable::getSymbolAttrName())) {
    segment.push_back('@');
    segment += escapePathComponent(symbol.getValue());
  }
  segment += "#" + std::to_string(ordinal);
  return segment;
}

void visitOperationChildren(Operation *parent, llvm::StringRef parentPath,
                            ModuleOp module, const ValidationPolicy &policy,
                            ValidationReport &report) {
  if (report.resourceLimitReported)
    return;
  for (auto regionEntry : llvm::enumerate(parent->getRegions())) {
    size_t regionIndex = regionEntry.index();
    Region &region = regionEntry.value();
    std::string regionPath =
        (llvm::Twine(parentPath) + "/region[" + llvm::Twine(regionIndex) + "]")
            .str();
    for (auto blockEntry : llvm::enumerate(region)) {
      size_t blockIndex = blockEntry.index();
      Block &block = blockEntry.value();
      std::string blockPath =
          (llvm::Twine(regionPath) + "/block[" + llvm::Twine(blockIndex) + "]")
              .str();
      ObjectTraversalState blockState;
      for (auto argumentEntry : llvm::enumerate(block.getArguments())) {
        visitType(argumentEntry.value().getType(), parentPath,
                  (llvm::Twine("region[") + llvm::Twine(regionIndex) +
                   "].block[" + llvm::Twine(blockIndex) + "].argument[" +
                   llvm::Twine(argumentEntry.index()) + "].type")
                      .str(),
                  argumentEntry.value().getLoc(), module, policy, report,
                  blockState);
        if (report.resourceLimitReported)
          return;
      }
      llvm::StringMap<unsigned> ordinals;
      for (Operation &operation : block) {
        llvm::StringRef operationName = operation.getName().getStringRef();
        unsigned ordinal = ordinals[operationName]++;
        std::string operationPath =
            blockPath + "/" + getOperationSegment(&operation, ordinal);
        llvm::StringRef dialect = operation.getName().getDialectNamespace();

        auto forbidden = policy.forbiddenDialects.find(dialect.str());
        if (forbidden != policy.forbiddenDialects.end()) {
          appendDiagnostic(policy, report,
                           makeTemplateDiagnostic(
                               policy, policy.forbiddenDialect, "operation",
                               operationName, operationPath,
                               getSourceLocation(operation.getLoc()), dialect));
        } else if (!isValidNamedDialectNamespace(dialect) ||
                   (dialect != "builtin" &&
                    !policy.allowedDialects.count(dialect.str()))) {
          appendDiagnostic(
              policy, report,
              makeTemplateDiagnostic(policy, policy.unknownDialect, "operation",
                                     operationName, operationPath,
                                     getSourceLocation(operation.getLoc()),
                                     dialect.empty() ? "<empty>" : dialect));
        }
        if (report.resourceLimitReported)
          return;

        visitOperationObjects(&operation, operationPath, module, policy,
                              report);
        if (report.resourceLimitReported)
          return;
        if (hasInvariant(policy, "linalg.no_unrealized_conversion_cast") ||
            hasInvariant(policy, "linalg.ranked_shaped_values") ||
            hasInvariant(policy, "linalg.generic_region_contract"))
          validateLinalgSemantics(&operation, operationPath, policy, report);
        if (report.resourceLimitReported)
          return;
        if (hasInvariant(policy, "gpu.tensor_encoding") ||
            hasInvariant(policy, "gpu.encoding_rank") ||
            hasInvariant(policy, "gpu.encoding_components") ||
            hasInvariant(policy, "gpu.shaped_element_type") ||
            hasInvariant(policy, "gpu.operation_contract") ||
            hasInvariant(policy, "gpu.dot_encoding_contract"))
          validateTritonGPUSemantics(&operation, operationPath, policy, report);
        if (report.resourceLimitReported)
          return;
        visitOperationChildren(&operation, operationPath, module, policy,
                               report);
        if (report.resourceLimitReported)
          return;
      }
    }
  }
}

DiagnosticCaptureResult
runWithCapturedDiagnostics(MLIRContext &context,
                           llvm::function_ref<LogicalResult()> action) {
  std::optional<CapturedMLIRDiagnostic> capturedError;
  ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diagnostic) {
    if (diagnostic.getSeverity() == DiagnosticSeverity::Error &&
        !capturedError) {
      capturedError =
          CapturedMLIRDiagnostic{sanitizeUTF8(diagnostic.str()),
                                 getSourceLocation(diagnostic.getLocation())};
    }
    return success();
  });
  LogicalResult result = action();
  return {result, std::move(capturedError)};
}

ValidationReport
makeFailureReport(const ValidationPolicy &policy,
                  const DiagnosticTemplate &diagnosticTemplate,
                  llvm::StringRef objectKind, llvm::StringRef objectName,
                  llvm::StringRef operationPath,
                  const std::optional<CapturedMLIRDiagnostic> &captured) {
  ValidationReport report{policy.specVersion, policy.track, policy.phase, {}};
  SourceLocation location;
  std::string detail;
  if (captured) {
    location = captured->location;
    detail = captured->message;
  }
  Diagnostic diagnostic =
      makeTemplateDiagnostic(policy, diagnosticTemplate, objectKind, objectName,
                             operationPath, std::move(location));
  if (!detail.empty()) {
    diagnostic.message += ": " + detail;
    diagnostic.message = boundDiagnosticField(std::move(diagnostic.message));
  }
  appendDiagnostic(policy, report, std::move(diagnostic));
  return report;
}

struct ParsedModule {
  OwningOpRef<ModuleOp> module;
  std::optional<ValidationReport> failure;
};

size_t skipWhitespace(llvm::StringRef text, size_t position) {
  while (position < text.size() &&
         llvm::isSpace(static_cast<unsigned char>(text[position])))
    ++position;
  return position;
}

bool isIdentifierCharacter(char character) {
  return llvm::isAlnum(static_cast<unsigned char>(character)) ||
         character == '_' || character == '$' || character == '.';
}

bool isPrefixedIdentifierSigil(char character) {
  return character == '#' || character == '!' || character == '@' ||
         character == '%' || character == '^';
}

size_t findUnquotedCharacter(llvm::StringRef text, size_t position,
                             char target) {
  bool inString = false;
  bool escaped = false;
  for (; position < text.size(); ++position) {
    char character = text[position];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        inString = false;
      }
      continue;
    }
    if (character == '"') {
      inString = true;
      continue;
    }
    if (character == target)
      return position;
  }
  return llvm::StringRef::npos;
}

bool denseLiteralContainsNonStringToken(llvm::StringRef literal) {
  bool inString = false;
  bool escaped = false;
  for (char character : literal) {
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        inString = false;
      }
      continue;
    }
    if (character == '"') {
      inString = true;
      continue;
    }
    if (llvm::isSpace(static_cast<unsigned char>(character)) ||
        character == '[' || character == ']' || character == '(' ||
        character == ')' || character == ',')
      continue;
    return true;
  }
  return false;
}

bool shapedTypeHasCustomElement(llvm::StringRef text, size_t bodyStart) {
  size_t position = skipWhitespace(text, bodyStart);
  if (position < text.size() && text[position] == '!')
    return true;

  unsigned nestedAngles = 0;
  for (; position < text.size(); ++position) {
    char character = text[position];
    if (character == '<') {
      ++nestedAngles;
      continue;
    }
    if (character == '>') {
      if (nestedAngles == 0)
        return false;
      --nestedAngles;
      continue;
    }
    if (character == ',' && nestedAngles == 0)
      return false;
    if (character != 'x' || nestedAngles != 0)
      continue;
    size_t elementStart = skipWhitespace(text, position + 1);
    if (elementStart < text.size() && text[elementStart] == '!')
      return true;
  }
  return false;
}

std::optional<size_t> findNextDenseToken(llvm::StringRef text,
                                         size_t searchFrom) {
  bool inString = false;
  bool escaped = false;
  bool inLineComment = false;
  for (size_t position = searchFrom; position < text.size(); ++position) {
    char character = text[position];
    if (inLineComment) {
      if (character == '\n' || character == '\r')
        inLineComment = false;
      continue;
    }
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        inString = false;
      }
      continue;
    }
    if (character == '"') {
      inString = true;
      continue;
    }
    if (character == '/' && position + 1 < text.size() &&
        text[position + 1] == '/') {
      inLineComment = true;
      ++position;
      continue;
    }
    if (!text.drop_front(position).starts_with("dense"))
      continue;

    size_t tokenEnd = position + 5;
    if ((position > 0 && (isIdentifierCharacter(text[position - 1]) ||
                          isPrefixedIdentifierSigil(text[position - 1]))) ||
        (tokenEnd < text.size() && isIdentifierCharacter(text[tokenEnd])))
      continue;
    return position;
  }
  return std::nullopt;
}

std::optional<size_t>
findUnsafeDenseCustomElementLiteral(llvm::StringRef text) {
  size_t searchFrom = 0;
  while (std::optional<size_t> dense = findNextDenseToken(text, searchFrom)) {
    size_t densePosition = *dense;
    searchFrom = densePosition + 5;

    size_t literalOpen = skipWhitespace(text, searchFrom);
    if (literalOpen >= text.size() || text[literalOpen] != '<')
      continue;
    size_t literalClose = findUnquotedCharacter(text, literalOpen + 1, '>');
    if (literalClose == llvm::StringRef::npos)
      continue;
    llvm::StringRef literal = text.slice(literalOpen + 1, literalClose);
    if (!denseLiteralContainsNonStringToken(literal))
      continue;

    size_t position = skipWhitespace(text, literalClose + 1);
    if (position >= text.size() || text[position] != ':')
      continue;
    position = skipWhitespace(text, position + 1);
    llvm::StringRef remainder = text.drop_front(position);
    size_t keywordLength = 0;
    if (remainder.starts_with("tensor"))
      keywordLength = 6;
    else if (remainder.starts_with("vector"))
      keywordLength = 6;
    else
      continue;
    position = skipWhitespace(text, position + keywordLength);
    if (position >= text.size() || text[position] != '<')
      continue;
    if (shapedTypeHasCustomElement(text, position + 1))
      return densePosition;
  }
  return std::nullopt;
}

std::optional<size_t> findNextUnquotedToken(llvm::StringRef text,
                                            size_t searchFrom,
                                            llvm::StringRef token) {
  bool inString = false;
  bool escaped = false;
  bool inLineComment = false;
  for (size_t position = searchFrom; position < text.size(); ++position) {
    char character = text[position];
    if (inLineComment) {
      if (character == '\n' || character == '\r')
        inLineComment = false;
      continue;
    }
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        inString = false;
      }
      continue;
    }
    if (character == '"') {
      inString = true;
      continue;
    }
    if (character == '/' && position + 1 < text.size() &&
        text[position + 1] == '/') {
      inLineComment = true;
      ++position;
      continue;
    }
    if (text.drop_front(position).starts_with(token))
      return position;
  }
  return std::nullopt;
}

std::optional<size_t> findMatchingAngle(llvm::StringRef text, size_t opening) {
  if (opening >= text.size() || text[opening] != '<')
    return std::nullopt;
  unsigned depth = 1;
  bool inString = false;
  bool escaped = false;
  for (size_t position = opening + 1; position < text.size(); ++position) {
    char character = text[position];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        inString = false;
      }
      continue;
    }
    if (character == '"') {
      inString = true;
      continue;
    }
    if (character == '<') {
      ++depth;
    } else if (character == '>' && --depth == 0) {
      return position;
    }
  }
  return std::nullopt;
}

std::optional<std::map<std::string, std::string>>
parseCustomAttributeDictionary(llvm::StringRef body) {
  body = body.trim();
  if (body.size() < 2 || body.front() != '{' || body.back() != '}')
    return std::nullopt;
  body = body.drop_front().drop_back();
  std::map<std::string, std::string> entries;
  size_t position = 0;
  while (true) {
    position = skipWhitespace(body, position);
    if (position == body.size())
      return entries;
    if (body[position] == ',') {
      ++position;
      position = skipWhitespace(body, position);
    }
    if (position == body.size())
      return std::nullopt;

    size_t keyStart = position;
    if (!llvm::isAlpha(static_cast<unsigned char>(body[position])) &&
        body[position] != '_')
      return std::nullopt;
    while (position < body.size() &&
           (llvm::isAlnum(static_cast<unsigned char>(body[position])) ||
            body[position] == '_' || body[position] == '-'))
      ++position;
    std::string key = body.slice(keyStart, position).str();
    position = skipWhitespace(body, position);
    if (position == body.size() || body[position] != '=')
      return std::nullopt;
    position = skipWhitespace(body, position + 1);
    size_t valueStart = position;
    int squareDepth = 0;
    int braceDepth = 0;
    int parenthesisDepth = 0;
    int angleDepth = 0;
    bool inString = false;
    bool escaped = false;
    for (; position < body.size(); ++position) {
      char character = body[position];
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (character == '\\') {
          escaped = true;
        } else if (character == '"') {
          inString = false;
        }
        continue;
      }
      if (character == '"') {
        inString = true;
        continue;
      }
      if (character == '[')
        ++squareDepth;
      else if (character == ']')
        --squareDepth;
      else if (character == '{')
        ++braceDepth;
      else if (character == '}')
        --braceDepth;
      else if (character == '(')
        ++parenthesisDepth;
      else if (character == ')')
        --parenthesisDepth;
      else if (character == '<')
        ++angleDepth;
      else if (character == '>')
        --angleDepth;
      else if (character == ',' && squareDepth == 0 && braceDepth == 0 &&
               parenthesisDepth == 0 && angleDepth == 0)
        break;
      if (squareDepth < 0 || braceDepth < 0 || parenthesisDepth < 0 ||
          angleDepth < 0)
        return std::nullopt;
    }
    llvm::StringRef value = body.slice(valueStart, position).trim();
    if (value.empty() || !entries.emplace(key, value.str()).second)
      return std::nullopt;
    if (position == body.size())
      return entries;
  }
}

bool hasExactlyKeys(const std::map<std::string, std::string> &entries,
                    const std::set<std::string> &required,
                    const std::set<std::string> &allowed) {
  for (const std::string &key : required)
    if (!entries.count(key))
      return false;
  for (const auto &entry : entries)
    if (!allowed.count(entry.first))
      return false;
  return true;
}

bool isUnsignedIntegerAttributeSpelling(llvm::StringRef value) {
  value = value.trim();
  size_t position = 0;
  while (position < value.size() &&
         llvm::isDigit(static_cast<unsigned char>(value[position])))
    ++position;
  if (position == 0)
    return false;
  position = skipWhitespace(value, position);
  if (position == value.size())
    return true;
  if (value[position] != ':')
    return false;
  position = skipWhitespace(value, position + 1);
  if (position == value.size() || value[position] != 'i')
    return false;
  ++position;
  size_t widthStart = position;
  while (position < value.size() &&
         llvm::isDigit(static_cast<unsigned char>(value[position])))
    ++position;
  return position > widthStart &&
         skipWhitespace(value, position) == value.size();
}

bool arrayHasExactlyElements(llvm::StringRef value, unsigned expected) {
  value = value.trim();
  if (value.size() < 2 || value.front() != '[' || value.back() != ']')
    return false;
  value = value.drop_front().drop_back().trim();
  if (value.empty())
    return expected == 0;
  unsigned elements = 1;
  int nestedDepth = 0;
  for (char character : value) {
    if (character == '[' || character == '(' || character == '{' ||
        character == '<')
      ++nestedDepth;
    else if (character == ']' || character == ')' || character == '}' ||
             character == '>')
      --nestedDepth;
    else if (character == ',' && nestedDepth == 0)
      ++elements;
    if (nestedDepth < 0)
      return false;
  }
  return nestedDepth == 0 && elements == expected;
}

std::optional<llvm::SmallVector<unsigned>>
parseUnsignedIntegerArraySpelling(llvm::StringRef value) {
  value = value.trim();
  if (value.size() < 2 || value.front() != '[' || value.back() != ']')
    return std::nullopt;
  value = value.drop_front().drop_back().trim();
  llvm::SmallVector<unsigned> result;
  if (value.empty())
    return result;

  llvm::SmallVector<llvm::StringRef> elements;
  value.split(elements, ',');
  for (llvm::StringRef element : elements) {
    element = element.trim();
    if (!isUnsignedIntegerAttributeSpelling(element))
      return std::nullopt;
    size_t digitEnd = 0;
    while (digitEnd < element.size() &&
           llvm::isDigit(static_cast<unsigned char>(element[digitEnd])))
      ++digitEnd;
    uint64_t parsed = 0;
    if (element.take_front(digitEnd).getAsInteger(10, parsed) ||
        parsed > std::numeric_limits<unsigned>::max())
      return std::nullopt;
    result.push_back(static_cast<unsigned>(parsed));
  }
  return result;
}

bool hasSafeExplicitCTALayout(
    const std::map<std::string, std::string> &entries) {
  constexpr llvm::StringLiteral kCTAsPerCGA = "CTAsPerCGA";
  constexpr llvm::StringLiteral kCTASplitNum = "CTASplitNum";
  constexpr llvm::StringLiteral kCTAOrder = "CTAOrder";
  bool hasCTAsPerCGA = entries.count(kCTAsPerCGA.str());
  bool hasCTASplitNum = entries.count(kCTASplitNum.str());
  bool hasCTAOrder = entries.count(kCTAOrder.str());
  if (!hasCTAsPerCGA && !hasCTASplitNum && !hasCTAOrder)
    return true;
  if (!hasCTAsPerCGA || !hasCTASplitNum || !hasCTAOrder)
    return false;

  auto ctasPerCGA =
      parseUnsignedIntegerArraySpelling(entries.at(kCTAsPerCGA.str()));
  auto ctaSplitNum =
      parseUnsignedIntegerArraySpelling(entries.at(kCTASplitNum.str()));
  auto ctaOrder =
      parseUnsignedIntegerArraySpelling(entries.at(kCTAOrder.str()));
  if (!ctasPerCGA || !ctaSplitNum || !ctaOrder ||
      ctasPerCGA->size() != ctaSplitNum->size() ||
      ctaSplitNum->size() != ctaOrder->size())
    return false;
  if (llvm::any_of(*ctasPerCGA, [](unsigned value) { return value == 0; }) ||
      llvm::any_of(*ctaSplitNum, [](unsigned value) { return value == 0; }))
    return false;

  llvm::SmallVector<bool> seen(ctaOrder->size(), false);
  for (unsigned value : *ctaOrder) {
    if (value >= seen.size() || seen[value])
      return false;
    seen[value] = true;
  }
  return true;
}

struct TextParserPreflightFailure {
  size_t offset;
  std::string message;
};

std::optional<TextParserPreflightFailure>
findUnsafeTritonGPUCTALayout(llvm::StringRef text) {
  constexpr llvm::StringLiteral dialectPrefix = "#triton_gpu.";
  size_t searchFrom = 0;
  while (std::optional<size_t> token =
             findNextUnquotedToken(text, searchFrom, dialectPrefix)) {
    size_t position = *token + dialectPrefix.size();
    while (position < text.size() &&
           (llvm::isAlnum(static_cast<unsigned char>(text[position])) ||
            text[position] == '_'))
      ++position;
    size_t opening = skipWhitespace(text, position);
    if (opening >= text.size() || text[opening] != '<') {
      searchFrom = position;
      continue;
    }
    std::optional<size_t> closing = findMatchingAngle(text, opening);
    if (!closing) {
      searchFrom = opening + 1;
      continue;
    }
    auto entries =
        parseCustomAttributeDictionary(text.slice(opening + 1, *closing));
    if (entries && !hasSafeExplicitCTALayout(*entries))
      return TextParserPreflightFailure{
          *token,
          "#triton_gpu custom attribute has an invalid explicit CTA layout",
      };
    searchFrom = *closing + 1;
  }
  return std::nullopt;
}

std::optional<TextParserPreflightFailure>
findOutOfRangeTritonGPUCustomInteger(llvm::StringRef text) {
  constexpr llvm::StringLiteral dialectToken = "#triton_gpu.";
  size_t searchFrom = 0;
  while (std::optional<size_t> token =
             findNextUnquotedToken(text, searchFrom, dialectToken)) {
    size_t position = *token + dialectToken.size();
    while (position < text.size() &&
           (llvm::isAlnum(static_cast<unsigned char>(text[position])) ||
            text[position] == '_'))
      ++position;
    size_t opening = skipWhitespace(text, position);
    if (opening >= text.size() || text[opening] != '<') {
      searchFrom = position;
      continue;
    }
    std::optional<size_t> closing = findMatchingAngle(text, opening);
    if (!closing) {
      searchFrom = opening + 1;
      continue;
    }

    bool inString = false;
    bool escaped = false;
    bool inLineComment = false;
    for (size_t index = opening + 1; index < *closing; ++index) {
      char character = text[index];
      if (inLineComment) {
        if (character == '\n' || character == '\r')
          inLineComment = false;
        continue;
      }
      if (inString) {
        if (escaped)
          escaped = false;
        else if (character == '\\')
          escaped = true;
        else if (character == '"')
          inString = false;
        continue;
      }
      if (character == '"') {
        inString = true;
        continue;
      }
      if (character == '/' && index + 1 < *closing &&
          text[index + 1] == '/') {
        inLineComment = true;
        ++index;
        continue;
      }
      if (!llvm::isDigit(static_cast<unsigned char>(character)))
        continue;
      if (index != opening + 1) {
        char previous = text[index - 1];
        if (llvm::isAlnum(static_cast<unsigned char>(previous)) ||
            previous == '_' || previous == '.')
          continue;
      }

      size_t valueEnd = index;
      unsigned radix = 10;
      if (text[index] == '0' && index + 2 < *closing &&
          (text[index + 1] == 'x' || text[index + 1] == 'X')) {
        radix = 16;
        valueEnd = index + 2;
        while (valueEnd < *closing &&
               llvm::isHexDigit(static_cast<unsigned char>(text[valueEnd])))
          ++valueEnd;
        if (valueEnd == index + 2)
          continue;
      } else {
        while (valueEnd < *closing &&
               llvm::isDigit(static_cast<unsigned char>(text[valueEnd])))
          ++valueEnd;
      }
      if (valueEnd < *closing) {
        char next = text[valueEnd];
        if (llvm::isAlnum(static_cast<unsigned char>(next)) || next == '_' ||
            next == '.') {
          index = valueEnd - 1;
          continue;
        }
      }

      llvm::StringRef spelling =
          radix == 16 ? text.slice(index + 2, valueEnd)
                      : text.slice(index, valueEnd);
      uint64_t value = 0;
      if (spelling.getAsInteger(radix, value) ||
          value > std::numeric_limits<unsigned>::max()) {
        return TextParserPreflightFailure{
            index,
            "#triton_gpu custom attribute integer exceeds the supported "
            "unsigned range",
        };
      }
      index = valueEnd - 1;
    }
    searchFrom = *closing + 1;
  }
  return std::nullopt;
}

std::optional<TextParserPreflightFailure>
findUnsafeTritonGPUCustomAttribute(llvm::StringRef text) {
  struct Contract {
    llvm::StringRef token;
    std::set<std::string> required;
    std::set<std::string> allowed;
  };
  const std::vector<Contract> contracts = {
      {
          "#triton_gpu.slice",
          {"dim", "parent"},
          {"dim", "parent"},
      },
      {
          "#triton_gpu.amd_mfma",
          {"versionMajor", "versionMinor", "warpsPerCTA", "instrShape",
           "isTransposed"},
          {"versionMajor", "versionMinor", "warpsPerCTA", "instrShape",
           "isTransposed", "CTAsPerCGA", "CTASplitNum", "CTAOrder"},
      },
  };
  for (const Contract &contract : contracts) {
    size_t searchFrom = 0;
    while (std::optional<size_t> token =
               findNextUnquotedToken(text, searchFrom, contract.token)) {
      searchFrom = *token + contract.token.size();
      size_t opening = skipWhitespace(text, searchFrom);
      if (opening >= text.size() || text[opening] != '<')
        continue;
      std::optional<size_t> closing = findMatchingAngle(text, opening);
      if (!closing)
        continue;
      auto entries =
          parseCustomAttributeDictionary(text.slice(opening + 1, *closing));
      bool safe = entries &&
                  hasExactlyKeys(*entries, contract.required, contract.allowed);
      if (safe && contract.token == "#triton_gpu.slice")
        safe = isUnsignedIntegerAttributeSpelling(entries->at("dim"));
      if (safe && contract.token == "#triton_gpu.amd_mfma")
        safe = arrayHasExactlyElements(entries->at("instrShape"), 2);
      if (!safe)
        return TextParserPreflightFailure{
            *token,
            contract.token.str() +
                " has missing, duplicate, unknown, or malformed fields",
        };
      searchFrom = *closing + 1;
    }
  }
  return std::nullopt;
}

SourceLocation getTextOffsetLocation(llvm::StringRef text, size_t offset,
                                     llvm::StringRef sourceName) {
  SourceLocation location;
  location.valid = true;
  location.file = sourceName.str();
  location.line = 1;
  location.column = 1;
  for (char character : text.take_front(offset)) {
    if (character == '\n') {
      ++location.line;
      location.column = 1;
    } else {
      ++location.column;
    }
  }
  return location;
}

std::optional<size_t> findTextStructuralNestingLimit(llvm::StringRef text) {
  std::vector<char> delimiters;
  bool inString = false;
  bool escaped = false;
  bool inLineComment = false;
  for (size_t position = 0; position < text.size(); ++position) {
    char character = text[position];
    if (inLineComment) {
      if (character == '\n' || character == '\r')
        inLineComment = false;
      continue;
    }
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        inString = false;
      }
      continue;
    }
    if (character == '"') {
      inString = true;
      continue;
    }
    if (character == '/' && position + 1 < text.size() &&
        text[position + 1] == '/') {
      inLineComment = true;
      ++position;
      continue;
    }
    bool opening = character == '{' || character == '[' || character == '(' ||
                   character == '<';
    if (opening) {
      if (delimiters.size() == kMaxAnchorIRStructuralDepth)
        return position;
      delimiters.push_back(character);
      continue;
    }

    // The '>' in MLIR's function-type and generic-op arrow is not a nesting
    // delimiter, including when the arrow itself occurs inside '<...>'.
    if (character == '>' && position != 0 && text[position - 1] == '-')
      continue;
    auto matches = [](char openingDelimiter, char closingDelimiter) {
      return (openingDelimiter == '{' && closingDelimiter == '}') ||
             (openingDelimiter == '[' && closingDelimiter == ']') ||
             (openingDelimiter == '(' && closingDelimiter == ')') ||
             (openingDelimiter == '<' && closingDelimiter == '>');
    };
    if (!delimiters.empty() && matches(delimiters.back(), character)) {
      delimiters.pop_back();
    }
  }
  return std::nullopt;
}

std::string maskLineCommentsPreservingOffsets(llvm::StringRef text) {
  std::string masked = text.str();
  bool inString = false;
  bool escaped = false;
  bool inLineComment = false;
  for (size_t position = 0; position < text.size(); ++position) {
    char character = text[position];
    if (inLineComment) {
      if (character == '\n' || character == '\r') {
        inLineComment = false;
      } else {
        masked[position] = ' ';
      }
      continue;
    }
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        inString = false;
      }
      continue;
    }
    if (character == '"') {
      inString = true;
      continue;
    }
    if (character == '/' && position + 1 < text.size() &&
        text[position + 1] == '/') {
      masked[position] = ' ';
      masked[position + 1] = ' ';
      inLineComment = true;
      ++position;
    }
  }
  return masked;
}

ParsedModule parseAnchorIRText(llvm::StringRef text, MLIRContext &context,
                               const ValidationPolicy &policy,
                               llvm::StringRef sourceName) {
  // All textual safety preflights must observe the same lexical view.  MLIR
  // accepts ``//`` comments inside custom-attribute dictionaries, but those
  // comments are not part of a key, integer, delimiter, or unsafe token.
  // Replace only comment bytes with spaces so scanners cannot reject legal
  // comments or let them bypass a safety contract, while source offsets remain
  // identical to the original text used for diagnostics and parsing.
  std::string maskedPreflightStorage =
      maskLineCommentsPreservingOffsets(text);
  llvm::StringRef preflightText(maskedPreflightStorage);

  // This runs before MLIR parses recursive Region syntax.  It also keeps a
  // textual request within the same nesting budget enforced for ModuleOp API
  // callers below.
  if (std::optional<size_t> limit =
          findTextStructuralNestingLimit(preflightText)) {
    return {
        nullptr,
        makeResourceLimitReport(
            policy, "text nesting depth",
            getTextOffsetLocation(text, *limit, sourceName)),
    };
  }
  // The pinned MLIR TensorLiteralParser asserts when a numeric dense literal
  // is paired with a custom element type.  Detect that invalid grammar before
  // entering the parser so native, Python and CLI text APIs all fail closed.
  if (std::optional<size_t> unsafe =
          findUnsafeDenseCustomElementLiteral(preflightText)) {
    CapturedMLIRDiagnostic diagnostic{
        "dense literal with a custom element type must use string elements",
        getTextOffsetLocation(text, *unsafe, sourceName)};
    return {
        nullptr,
        makeFailureReport(policy, policy.parseFailure, "module",
                          "builtin.module", "", diagnostic),
    };
  }
  if (std::optional<TextParserPreflightFailure> unsafe =
          findOutOfRangeTritonGPUCustomInteger(preflightText)) {
    CapturedMLIRDiagnostic diagnostic{
        unsafe->message,
        getTextOffsetLocation(text, unsafe->offset, sourceName)};
    return {
        nullptr,
        makeFailureReport(policy, policy.parseFailure, "module",
                          "builtin.module", "", diagnostic),
    };
  }
  if (std::optional<TextParserPreflightFailure> unsafe =
          findUnsafeTritonGPUCTALayout(preflightText)) {
    CapturedMLIRDiagnostic diagnostic{
        unsafe->message,
        getTextOffsetLocation(text, unsafe->offset, sourceName)};
    return {
        nullptr,
        makeFailureReport(policy, policy.parseFailure, "module",
                          "builtin.module", "", diagnostic),
    };
  }
  if (std::optional<TextParserPreflightFailure> unsafe =
          findUnsafeTritonGPUCustomAttribute(preflightText)) {
    CapturedMLIRDiagnostic diagnostic{
        unsafe->message,
        getTextOffsetLocation(text, unsafe->offset, sourceName)};
    return {
        nullptr,
        makeFailureReport(policy, policy.parseFailure, "module",
                          "builtin.module", "", diagnostic),
    };
  }

  OwningOpRef<ModuleOp> module;
  DiagnosticCaptureResult parser = runWithCapturedDiagnostics(context, [&]() {
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    module = parseSourceString<ModuleOp>(text, config, sourceName);
    return success(static_cast<bool>(module));
  });
  if (failed(parser.result) || !module)
    return {
        nullptr,
        makeFailureReport(policy, policy.parseFailure, "module",
                          "builtin.module", "", parser.error),
    };
  return {std::move(module), std::nullopt};
}

} // namespace

ValidationReport validateAnchorIR(ModuleOp module,
                                  const ValidationPolicy &policy) {
  if (std::optional<StructuralLimit> limit =
          findOperationStructuralLimit(module)) {
    llvm::StringRef objectName =
        limit->kind == StructuralLimitKind::OperationDepth
            ? "operation nesting depth"
            : "operation count";
    return makeResourceLimitReport(policy, objectName,
                                   getSourceLocation(module.getLoc()));
  }
  ValidationReport report{policy.specVersion, policy.track, policy.phase, {}};
  // Policy and AnchorIR semantic checks deliberately run before the upstream
  // verifier.  Several verifiers in the pinned Triton/MLIR revision assume
  // layout interfaces or module attributes are already valid and may abort
  // instead of returning failure.  Invalid policy objects must therefore
  // become stable AIR-* diagnostics before those unsafe code paths are entered.
  visitOperationObjects(module.getOperation(), "builtin.module", module, policy,
                        report);
  visitOperationChildren(module.getOperation(), "builtin.module", module,
                         policy, report);

  // The configuration scan recursively walks both the operation tree and
  // nested Type/Attribute subelements.  Enter it only after the explicitly
  // bounded traversal above has established that those structures are safe.
  bool verifierSafe = !report.resourceLimitReported;
  if (verifierSafe)
    verifierSafe = validateGPUConfiguration(module, policy, report);

  verifierSafe = verifierSafe && report.valid();
  if (verifierSafe && !isTritonDotVerifierSafe(module)) {
    // A verifier safety preflight is a rejection, never permission to skip the
    // verifier and return a valid report.  The official policy diagnoses these
    // inputs through gpu.dot_encoding_contract first; this fallback also keeps
    // the public native API fail-closed if a caller supplies a reduced policy.
    CapturedMLIRDiagnostic unsafe{
        "MLIR verifier safety preflight rejected an unsafe tt.dot layout",
        getSourceLocation(module.getLoc())};
    ValidationReport verifierReport =
        makeFailureReport(policy, policy.verifyFailure, "module",
                          "builtin.module", "builtin.module", unsafe);
    appendDiagnostic(policy, report,
                     std::move(verifierReport.diagnostics.front()));
    verifierSafe = false;
  }
  if (verifierSafe && !isTritonReshapeVerifierSafe(module)) {
    CapturedMLIRDiagnostic unsafe{
        "MLIR verifier safety preflight rejected malformed tt.reshape "
        "cardinality",
        getSourceLocation(module.getLoc())};
    ValidationReport verifierReport =
        makeFailureReport(policy, policy.verifyFailure, "module",
                          "builtin.module", "builtin.module", unsafe);
    appendDiagnostic(policy, report,
                     std::move(verifierReport.diagnostics.front()));
    verifierSafe = false;
  }
  if (verifierSafe) {
    DiagnosticCaptureResult verifier = runWithCapturedDiagnostics(
        *module.getContext(), [&]() { return verify(module); });
    if (failed(verifier.result)) {
      ValidationReport verifierReport =
          makeFailureReport(policy, policy.verifyFailure, "module",
                            "builtin.module", "builtin.module", verifier.error);
      appendDiagnostic(policy, report,
                       std::move(verifierReport.diagnostics.front()));
    }
  }
  return report;
}

ValidationReport validateAnchorIRText(llvm::StringRef text,
                                      MLIRContext &context,
                                      const ValidationPolicy &policy,
                                      llvm::StringRef sourceName) {
  auto contextMutex = getAnchorIRTextContextMutex(context);
  std::lock_guard<std::recursive_mutex> contextGuard(*contextMutex);
  bool previouslyAllowed = context.allowsUnregisteredDialects();
  context.allowUnregisteredDialects(true);
  auto restoreUnregisteredDialects = llvm::make_scope_exit(
      [&]() { context.allowUnregisteredDialects(previouslyAllowed); });

  ParsedModule parsed = parseAnchorIRText(text, context, policy, sourceName);
  if (parsed.failure)
    return std::move(*parsed.failure);
  return validateAnchorIR(*parsed.module, policy);
}

NormalizationResult normalizeAnchorIR(ModuleOp module,
                                      const ValidationPolicy &policy) {
  ValidationReport validation = validateAnchorIR(module, policy);
  if (!validation.valid())
    return {std::move(validation), std::nullopt};
  return {std::move(validation), printNormalizedModule(module)};
}

NormalizationResult normalizeAnchorIRText(llvm::StringRef text,
                                          MLIRContext &context,
                                          const ValidationPolicy &policy,
                                          llvm::StringRef sourceName) {
  auto contextMutex = getAnchorIRTextContextMutex(context);
  std::lock_guard<std::recursive_mutex> contextGuard(*contextMutex);
  bool previouslyAllowed = context.allowsUnregisteredDialects();
  context.allowUnregisteredDialects(true);
  auto restoreUnregisteredDialects = llvm::make_scope_exit(
      [&]() { context.allowUnregisteredDialects(previouslyAllowed); });

  ParsedModule parsed = parseAnchorIRText(text, context, policy, sourceName);
  if (parsed.failure)
    return {std::move(*parsed.failure), std::nullopt};
  ValidationReport validation = validateAnchorIR(*parsed.module, policy);
  if (!validation.valid())
    return {std::move(validation), std::nullopt};
  return {std::move(validation), printNormalizedModule(*parsed.module)};
}

} // namespace mlir::triton::anchor
