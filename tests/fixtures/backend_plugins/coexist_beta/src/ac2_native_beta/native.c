#if defined(__GNUC__)
#define AC2_EXPORT __attribute__((visibility("default")))
#else
#define AC2_EXPORT
#endif

AC2_EXPORT int triton_anchor_ac2_native_beta_symbol(void) {
  return 22;
}
