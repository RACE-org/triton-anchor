/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonGPUOps.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace triton {
namespace gpu {

class TritonGPUDialect : public ::mlir::Dialect {
  explicit TritonGPUDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~TritonGPUDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("triton_gpu");
  }

  /// Parse an attribute registered to this dialect.
  ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                   ::mlir::Type type) const override;

  /// Print an attribute registered to this dialect.
  void printAttribute(::mlir::Attribute attr,
                      ::mlir::DialectAsmPrinter &os) const override;

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;

    /// Provides a hook for verifying dialect attributes attached to the given
    /// op.
    ::mlir::LogicalResult verifyOperationAttribute(
        ::mlir::Operation *op, ::mlir::NamedAttribute attribute) override;

    static std::string getNumWarpsAttrName() { return "triton_gpu.num-warps"; }
    static int getNumWarps(ModuleOp mod) {
      if (!mod->hasAttr("triton_gpu.num-warps"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu.num-warps attribute");
      return cast<IntegerAttr>(mod->getAttr("triton_gpu.num-warps")).getInt();
    }
    static int getNumCTAs(ModuleOp mod) {
      if (!mod->hasAttr("triton_gpu.num-ctas"))
        return 1;
      return cast<IntegerAttr>(mod->getAttr("triton_gpu.num-ctas")).getInt();
    }
    void registerTypes();

    static std::string getThreadsPerWarpAttrName() { return "triton_gpu.threads-per-warp"; }

    static int getThreadsPerWarp(ModuleOp mod) {
      Attribute threadsPerWarp = mod->getDiscardableAttr("triton_gpu.threads-per-warp");
      if(!threadsPerWarp) {
        return 32;
      }
      return cast<IntegerAttr>(threadsPerWarp).getInt();
    }
  };
} // namespace gpu
} // namespace triton
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::TritonGPUDialect)
