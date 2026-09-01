module attributes {
  "ttg.num-warps" = 1 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  "ttg.num-ctas" = 1 : i32
} {
  %scalar = arith.constant 1 : i32
  %tensor = tt.splat %scalar : i32
      -> tensor<4x4xi32, #ttg.blocked<{
          sizePerThread = [1],
          threadsPerWarp = [32],
          warpsPerCTA = [1],
          order = [0]
        }>>
}
