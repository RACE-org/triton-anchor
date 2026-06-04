#include "PrintOpToSPIRV.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct PrintOpConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::PrintOp> {
public:
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::PrintOp>::ConvertTritonGPUOpToSPIRVPattern;
  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    auto getPid = [&](mlir::gpu::Dimension axis) {
      return getProgramId(rewriter, loc, axis);
    };
    std::array<Value, 3> pid = {getPid(mlir::gpu::Dimension::x),
                                getPid(mlir::gpu::Dimension::y),
                                getPid(mlir::gpu::Dimension::z)};

    // Simple printf of a string without any tensors.
    if (op.getNumOperands() == 0) {
      std::string formatStr;
      llvm::raw_string_ostream os(formatStr);
      os << "pid (" << getFormatSubstr(pid[0]) << ", "
         << getFormatSubstr(pid[1]) << ", " << getFormatSubstr(pid[2]) << ")"
         << op.getPrefix();
      spirvPrintf(formatStr, {pid[0], pid[1], pid[2]}, 0, rewriter);
      rewriter.eraseOp(op);
      return success();
    }
    for (size_t i = 0; i < op.getNumOperands(); i++) {
      auto elems = getTypeConverter()->unpackLLElements(
          loc, adaptor.getOperands()[i], rewriter,
          op.getOperands()[i].getType());
      SmallVector<int, 8> dimWidths;
      SmallVector<SmallVector<Value>> indices;
      if (auto rankedTy =
              dyn_cast<RankedTensorType>(op.getOperand(i).getType())) {
        indices =
            emitIndices(loc, rewriter, rankedTy.getEncoding(), rankedTy, true);
        for (int64_t dim : rankedTy.getShape()) {
          if (dim > 0) {
            dimWidths.push_back(static_cast<int>(std::ceil(std::log10(dim))));
          } else {
            dimWidths.push_back(0);
          }
        }
      } else {
        assert(elems.size() == 1);
        indices.push_back({});
      }
      if (!elems.empty()) {
        printTensor(op.getPrefix(), /*operand=*/i,
                    /*numOperands=*/op.getNumOperands(), elems, pid, indices,
                    dimWidths, op.getHex(), rewriter);
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  Value printfBoolValue(ConversionPatternRewriter &rewriter,
                        Value value) const {
    auto *context = rewriter.getContext();
    auto loc = UnknownLoc::get(context);
    auto type = value.getType();
    if (type.isIntOrFloat() && type.getIntOrFloatBitWidth() == 1) {
      return select(value, i32_val(0xFFFFFFFF), i32_val(0));
    }
    return value;
  }

  void printTensor(StringRef prefixStr, size_t operand, size_t numOperands,
                   ArrayRef<Value> elems, std::array<Value, 3> pid,
                   ArrayRef<SmallVector<Value>> indices,
                   ArrayRef<int> dimWidths, bool hex,
                   ConversionPatternRewriter &rewriter) const {
    assert(!elems.empty());
    assert(elems.size() == indices.size());
    assert(dimWidths.size() == indices.front().size());
    size_t rank = dimWidths.size();
    Value formatStrValue;
    int formatStrByteCount = 0;
    for (int i = 0; i < elems.size(); i++) {
      std::string formatStr;
      llvm::raw_string_ostream os(formatStr);
      constexpr int kMaxPrintfOperands = 32;
      SmallVector<Value, kMaxPrintfOperands> printfOperands;
      os << "pid (";
      for (int j = 0; j < pid.size(); j++) {
        if (j != 0) {
          os << ", ";
        }
        os << getFormatSubstr(pid[j]);
        printfOperands.push_back(pid[j]);
      }
      os << ") ";
      int maxAllowedRank = kMaxPrintfOperands - printfOperands.size() - 2;

      os << "idx (";
      const auto &index = indices[i];
      for (size_t dim = 0; dim < index.size(); dim++) {
        if (dim != 0) {
          os << ", ";
        }
        if (dim == maxAllowedRank) {
          os << "... (truncated)";
          break;
        }
        os << getFormatSubstr(index[dim], /*hex=*/false,
                              /*width=*/dimWidths[dim]);
        printfOperands.push_back(index[dim]);
      }
      os << ")" << prefixStr;

      if (numOperands > 1) {
        os << "(operand " << operand << ") ";
      }

      auto elem = elems[i];
      os << getFormatSubstr(elem, hex);
      printfOperands.push_back(elem);

      // process bool value
      SmallVector<Value, 16> newArgs;
      for (auto arg : printfOperands) {
        Value newArg = printfBoolValue(rewriter, arg);
        newArgs.push_back(newArg);
      }
      // It's the same format string each iteration, but it's a lot easier
      //  if we construct the format string at the same time as we populate
      // printfOperands.  But we don't want to create BLOCK_SIZE duplicate
      // strings, so we cache the Value.
      if (i == 0) {
        formatStrValue = spirvPrintf(formatStr, newArgs, operand, rewriter);
      } else {
        rewriter.create<mlir::spirv::CLPrintfOp>(
            UnknownLoc::get(rewriter.getContext()), i32_ty, formatStrValue,
            newArgs);
      }
    }
  }

  Value spirvPrintf(StringRef msg, ValueRange arg, size_t i,
                    ConversionPatternRewriter &rewriter) const {
    assert(!msg.empty() && "printf with empty string not supported");
    llvm::SmallString<64> msgNewline(msg);
    msgNewline.push_back('\n');
    msgNewline.push_back('\0');
    std::string key = "print_" + std::to_string(i);
    Value messageString = spirv::addStringToModule(
        UnknownLoc::get(rewriter.getContext()), rewriter, key, msgNewline);
    rewriter.create<mlir::spirv::CLPrintfOp>(
        UnknownLoc::get(rewriter.getContext()), i32_ty, messageString, arg);
    return messageString;
  }

  std::string getFormatSubstr(Value value, bool hex = false,
                              std::optional<int> width = std::nullopt) const {
    Type type = value.getType();
    if (isa<spirv::PointerType>(type)) {
      return "%p";
    }
    // Hex is "0x%0nx" or "0x%0nllx", where n is the number of hex digits in
    // the type (so 4 for fp16, 8 for int32, 16 for int64).
    if (hex) {
      // Ignore `width` for `hex` values, pad to typeWidth.
      std::string ret =
          "0x%0" + std::to_string(type.getIntOrFloatBitWidth() / 4);
      if (type.getIntOrFloatBitWidth() > 32) {
        ret += "ll";
      }
      ret += "x";
      return ret;
    }

    std::string prefix = "%";
    if (width.has_value()) {
      prefix += std::to_string(*width);
    } else if (hex) {
      prefix += "0";
      prefix += std::to_string(value.getType().getIntOrFloatBitWidth() / 4);
    }

    if (type.isBF16() || type.isF16() || type.isF32() || type.isF64()) {
      return prefix + "f";
    } else if (type.isSignedInteger()) {
      if (type.getIntOrFloatBitWidth() == 64)
        return prefix + "li";
      else
        return prefix + "i";
    } else if (type.isUnsignedInteger() || type.isSignlessInteger()) {
      if (type.getIntOrFloatBitWidth() == 64)
        return prefix + "lu";
      else
        return prefix + "u";
    }
    assert(false && "not supported type");
    return "";
  }
};

} // namespace
void populatePrintOpToSPIRVPattern(TritonGPUToSPIRVTypeConverter &typeConverter,
                                   mlir::MLIRContext *context,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit) {
  patterns.add<PrintOpConversion>(typeConverter, context, benefit);
}
