#include "AnchorIRValidatorBindings.h"
#include "triton-anchor/Validation/AnchorIRValidator.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

mlir::triton::anchor::DiagnosticTemplate parseTemplate(const py::dict &value) {
  return {
      py::cast<std::string>(value["code"]),
      py::cast<std::string>(value["message"]),
      py::cast<std::string>(value["hint"]),
  };
}

mlir::triton::anchor::ValidationPolicy parsePolicy(const py::dict &value) {
  mlir::triton::anchor::ValidationPolicy policy;
  policy.specVersion = py::cast<std::string>(value["spec_version"]);
  policy.track = py::cast<std::string>(value["track"]);
  policy.phase = py::cast<std::string>(value["phase"]);
  auto allowed = py::cast<std::vector<std::string>>(value["allowed_dialects"]);
  auto forbidden =
      py::cast<std::vector<std::string>>(value["forbidden_dialects"]);
  if (value.contains("core_allowed_dialects")) {
    auto coreAllowed =
        py::cast<std::vector<std::string>>(value["core_allowed_dialects"]);
    policy.coreAllowedDialects.insert(coreAllowed.begin(), coreAllowed.end());
  } else {
    policy.coreAllowedDialects.insert(allowed.begin(), allowed.end());
  }
  policy.allowedDialects.insert(allowed.begin(), allowed.end());
  policy.forbiddenDialects.insert(forbidden.begin(), forbidden.end());
  if (value.contains("extension_dialects")) {
    auto extensions =
        py::cast<std::vector<std::string>>(value["extension_dialects"]);
    policy.extensionDialects.insert(extensions.begin(), extensions.end());
  }
  if (!policy.extensionDialects.empty() && policy.phase != "post_hook")
    throw py::value_error(
        "extension_dialects are only valid for post_hook validation");
  for (const std::string &extension : policy.extensionDialects) {
    if (policy.forbiddenDialects.count(extension))
      throw py::value_error(
          "extension_dialects cannot include core-forbidden dialect: " +
          extension);
    if (policy.coreAllowedDialects.count(extension))
      throw py::value_error(
          "extension_dialects cannot redeclare core dialect: " + extension);
    if (!policy.allowedDialects.count(extension))
      throw py::value_error(
          "extension_dialects must be included in allowed_dialects: " +
          extension);
  }
  if (value.contains("enabled_invariants")) {
    auto invariants =
        py::cast<std::vector<std::string>>(value["enabled_invariants"]);
    policy.enabledInvariants.insert(invariants.begin(), invariants.end());
  }
  if (value.contains("semantic_diagnostics")) {
    auto diagnostics = py::cast<py::dict>(value["semantic_diagnostics"]);
    for (auto item : diagnostics)
      policy.semanticDiagnostics.emplace(
          py::cast<std::string>(item.first),
          parseTemplate(py::cast<py::dict>(item.second)));
  }
  for (const std::string &invariant : policy.enabledInvariants) {
    if (!policy.semanticDiagnostics.count(invariant))
      throw py::value_error("enabled invariant has no semantic diagnostic: " +
                            invariant);
  }
  policy.unknownDialect =
      parseTemplate(py::cast<py::dict>(value["unknown_dialect_diagnostic"]));
  policy.forbiddenDialect =
      parseTemplate(py::cast<py::dict>(value["forbidden_dialect_diagnostic"]));
  policy.unknownType =
      parseTemplate(py::cast<py::dict>(value["unknown_type_diagnostic"]));
  policy.forbiddenType =
      parseTemplate(py::cast<py::dict>(value["forbidden_type_diagnostic"]));
  policy.unknownAttribute =
      parseTemplate(py::cast<py::dict>(value["unknown_attribute_diagnostic"]));
  policy.forbiddenAttribute = parseTemplate(
      py::cast<py::dict>(value["forbidden_attribute_diagnostic"]));
  policy.resourceLimit =
      parseTemplate(py::cast<py::dict>(value["resource_limit_diagnostic"]));
  policy.parseFailure =
      parseTemplate(py::cast<py::dict>(value["parse_failure_diagnostic"]));
  policy.verifyFailure =
      parseTemplate(py::cast<py::dict>(value["verify_failure_diagnostic"]));
  return policy;
}

py::object
locationToPython(const mlir::triton::anchor::SourceLocation &location) {
  if (!location.valid)
    return py::none();
  py::dict result;
  result["file"] = location.file;
  result["line"] = location.line;
  result["column"] = location.column;
  return std::move(result);
}

py::dict reportToPython(const mlir::triton::anchor::ValidationReport &report) {
  py::list diagnostics;
  for (const auto &diagnostic : report.diagnostics) {
    py::dict item;
    item["code"] = diagnostic.code;
    item["severity"] = diagnostic.severity;
    item["message"] = diagnostic.message;
    item["hint"] = diagnostic.hint;
    item["spec_version"] = diagnostic.specVersion;
    item["track"] = diagnostic.track;
    item["phase"] = diagnostic.phase;
    item["object_kind"] = diagnostic.objectKind;
    item["object_name"] = diagnostic.objectName;
    item["operation_path"] = diagnostic.operationPath;
    item["object_path"] = diagnostic.objectPath;
    item["location"] = locationToPython(diagnostic.location);
    diagnostics.append(std::move(item));
  }

  py::dict result;
  result["valid"] = report.valid();
  result["spec_version"] = report.specVersion;
  result["track"] = report.track;
  result["phase"] = report.phase;
  result["diagnostics"] = std::move(diagnostics);
  return result;
}

py::dict
normalizationToPython(const mlir::triton::anchor::NormalizationResult &result) {
  py::dict output;
  output["validation_report"] = reportToPython(result.validation);
  if (result.normalizedText)
    output["normalized_text"] = *result.normalizedText;
  else
    output["normalized_text"] = py::none();
  return output;
}

} // namespace

void init_triton_anchor_validator(py::module_ &module) {
  module.def(
      "check_anchor_ir_module_context",
      [](mlir::ModuleOp &irModule, mlir::MLIRContext &context) {
        if (irModule.getContext() != &context)
          throw py::value_error(
              "context does not own the supplied AnchorIR module");
      },
      py::arg("module"), py::arg("context"),
      "Check that a Python context owns an AnchorIR ModuleOp.");

  module.def(
      "clone_anchor_ir_module",
      [](mlir::ModuleOp &irModule, mlir::MLIRContext &context) {
        if (irModule.getContext() != &context)
          throw py::value_error(
              "context does not own the AnchorIR module being cloned");
        return irModule.clone();
      },
      py::arg("module"), py::arg("context"),
      py::return_value_policy::take_ownership, py::keep_alive<0, 2>(),
      "Clone a validated ModuleOp and retain its owning MLIR context.");

  module.def(
      "validate_anchor_ir",
      [](mlir::ModuleOp &irModule, const py::dict &policy) {
        return reportToPython(mlir::triton::anchor::validateAnchorIR(
            irModule, parsePolicy(policy)));
      },
      py::arg("module"), py::arg("policy"));

  module.def(
      "validate_anchor_ir_text",
      [](const std::string &text, mlir::MLIRContext &context,
         const py::dict &policy, const std::string &sourceName) {
        auto parsedPolicy = parsePolicy(policy);
        mlir::triton::anchor::ValidationReport report;
        {
          py::gil_scoped_release release;
          report = mlir::triton::anchor::validateAnchorIRText(
              text, context, parsedPolicy, sourceName);
        }
        return reportToPython(report);
      },
      py::arg("text"), py::arg("context"), py::arg("policy"),
      py::arg("source_name") = "<anchor-ir>");

  module.def(
      "normalize_anchor_ir",
      [](mlir::ModuleOp &irModule, const py::dict &policy) {
        return normalizationToPython(mlir::triton::anchor::normalizeAnchorIR(
            irModule, parsePolicy(policy)));
      },
      py::arg("module"), py::arg("policy"));

  module.def(
      "normalize_anchor_ir_text",
      [](const std::string &text, mlir::MLIRContext &context,
         const py::dict &policy, const std::string &sourceName) {
        auto parsedPolicy = parsePolicy(policy);
        mlir::triton::anchor::NormalizationResult result;
        {
          py::gil_scoped_release release;
          result = mlir::triton::anchor::normalizeAnchorIRText(
              text, context, parsedPolicy, sourceName);
        }
        return normalizationToPython(result);
      },
      py::arg("text"), py::arg("context"), py::arg("policy"),
      py::arg("source_name") = "<anchor-ir>");
}
