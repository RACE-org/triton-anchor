module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  tt.make_range {start = 0 : i32, end = 16 : i32}
      : tensor<16xi32, #triton_gpu.blocked<{
          sizePerThread = [1],
          threadsPerWarp = [32],
          warpsPerCTA = [1],
          order = [0]
        }>>
}
