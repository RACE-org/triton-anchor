# triton-shared Conversion passes
# 来源: FlagTree/third_party/tsingmicro/lib/Conversion/ 中的 triton-shared 相关模块
#
# 说明：
#   以下独立 pass 不依赖硬件专有 Dialect，可以在 triton-anchor 中编译。
#   TritonToCoreDialects 是管线编排 pass，依赖了 MagicKernel/Address 等硬件 Dialect，
#   因此保留在 triton-tsingmicro-backend 中编译。

add_subdirectory(TritonArithToLinalg)
add_subdirectory(StructuredToMemref)
add_subdirectory(ConvertTritonPtr)
add_subdirectory(ReconcilePtrCasts)

# TritonToCoreDialects 依赖硬件 Dialect（MagicKernel, Address 等），
# 在 triton-tsingmicro-backend 中编译，此处仅保留头文件
# add_subdirectory(TritonToCoreDialects)

# 以下模块在 FlagTree 中已注释掉（flir 遗留），暂不编译
# add_subdirectory(TritonPtrToMemref)
