#if defined(__GNUC__)
#define W10_EXPORT __attribute__((visibility("default")))
#else
#define W10_EXPORT
#endif

W10_EXPORT int triton_anchor_w10_native_collision_symbol(void) {
  return 2;
}
