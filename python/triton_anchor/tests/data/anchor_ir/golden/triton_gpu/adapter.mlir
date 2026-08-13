module attributes {
  "ttg.num-warps" = 1 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  "ttg.num-ctas" = 1 : i32
} {
  tt.make_range {start = 0 : i32, end = 16 : i32}
      : tensor<16xi32, #ttg.blocked<{
          sizePerThread = [1],
          threadsPerWarp = [32],
          warpsPerCTA = [1],
          order = [0]
        }>>
}
