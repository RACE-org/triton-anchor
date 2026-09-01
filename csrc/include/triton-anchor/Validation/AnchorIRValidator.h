#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#if defined(_WIN32)
#if defined(TRITON_ANCHOR_VALIDATION_EXPORTS)
#define TRITON_ANCHOR_VALIDATION_API __declspec(dllexport)
#else
#define TRITON_ANCHOR_VALIDATION_API __declspec(dllimport)
#endif
#else
#define TRITON_ANCHOR_VALIDATION_API __attribute__((visibility("default")))
#endif

namespace mlir::triton::anchor {

struct DiagnosticTemplate {
  std::string code;
  std::string message;
  std::string hint;
};

struct ValidationPolicy {
  std::string specVersion;
  std::string track;
  std::string phase;
  std::set<std::string> coreAllowedDialects;
  std::set<std::string> allowedDialects;
  std::set<std::string> extensionDialects;
  std::set<std::string> forbiddenDialects;
  std::set<std::string> enabledInvariants;
  std::map<std::string, DiagnosticTemplate> semanticDiagnostics;
  DiagnosticTemplate unknownDialect;
  DiagnosticTemplate forbiddenDialect;
  DiagnosticTemplate unknownType;
  DiagnosticTemplate forbiddenType;
  DiagnosticTemplate unknownAttribute;
  DiagnosticTemplate forbiddenAttribute;
  DiagnosticTemplate resourceLimit;
  DiagnosticTemplate parseFailure;
  DiagnosticTemplate verifyFailure;
};

struct SourceLocation {
  bool valid = false;
  std::string file;
  int64_t line = 0;
  int64_t column = 0;
};

struct Diagnostic {
  std::string code;
  std::string severity = "error";
  std::string message;
  std::string hint;
  std::string specVersion;
  std::string track;
  std::string phase;
  std::string objectKind;
  std::string objectName;
  std::string operationPath;
  std::string objectPath;
  SourceLocation location;
};

struct ValidationReport {
  std::string specVersion;
  std::string track;
  std::string phase;
  std::vector<Diagnostic> diagnostics;
  // Internal traversal/diagnostic budgeting state.  It is deliberately not
  // part of the Python report schema.
  bool resourceLimitReported = false;
  // Count logical Type/Attribute visits across the whole ModuleOp.  MLIR
  // uniquing allows a shallow alias DAG to share one child from many parents;
  // an ancestor-only cycle guard would otherwise walk that DAG exponentially.
  size_t objectVisitCount = 0;

  bool valid() const { return diagnostics.empty(); }
};

struct NormalizationResult {
  ValidationReport validation;
  std::optional<std::string> normalizedText;
};

/// Validate an already parsed, real ModuleOp. Policy and AnchorIR semantic
/// traversal run before the MLIR verifier so malformed objects cannot enter
/// verifier paths known to abort in the pinned dependency revision. A clean
/// preflight is still followed by the normal MLIR verifier.
TRITON_ANCHOR_VALIDATION_API ValidationReport
validateAnchorIR(ModuleOp module, const ValidationPolicy &policy);

/// Parse text without implicit verification, report syntax failure separately,
/// then invoke the same ModuleOp validator used by validateAnchorIR().
TRITON_ANCHOR_VALIDATION_API ValidationReport
validateAnchorIRText(llvm::StringRef text, MLIRContext &context,
                     const ValidationPolicy &policy,
                     llvm::StringRef sourceName);

/// Validate and then print a deterministic, location-free generic MLIR form.
/// Invalid IR never receives normalized text.
TRITON_ANCHOR_VALIDATION_API NormalizationResult
normalizeAnchorIR(ModuleOp module, const ValidationPolicy &policy);

/// Text variant with the same validation gate as normalizeAnchorIR().
TRITON_ANCHOR_VALIDATION_API NormalizationResult
normalizeAnchorIRText(llvm::StringRef text, MLIRContext &context,
                      const ValidationPolicy &policy,
                      llvm::StringRef sourceName);

} // namespace mlir::triton::anchor

#undef TRITON_ANCHOR_VALIDATION_API
