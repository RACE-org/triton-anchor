#include <pybind11/pybind11.h>
namespace py = pybind11;

// ═══════════════════════════════════════════════════════════════════════
// 后端插件初始化宏（仅在 TRITON_HAS_BACKENDS 定义时启用）
// ═══════════════════════════════════════════════════════════════════════
#ifdef TRITON_HAS_BACKENDS

#define FOR_EACH_1(MACRO, X) MACRO(X)
#define FOR_EACH_2(MACRO, X, ...) MACRO(X) FOR_EACH_1(MACRO, __VA_ARGS__)
#define FOR_EACH_3(MACRO, X, ...) MACRO(X) FOR_EACH_2(MACRO, __VA_ARGS__)
#define FOR_EACH_4(MACRO, X, ...) MACRO(X) FOR_EACH_3(MACRO, __VA_ARGS__)

#define FOR_EACH_NARG(...) FOR_EACH_NARG_(__VA_ARGS__, FOR_EACH_RSEQ_N())
#define FOR_EACH_NARG_(...) FOR_EACH_ARG_N(__VA_ARGS__)
#define FOR_EACH_ARG_N(_1, _2, _3, _4, N, ...) N
#define FOR_EACH_RSEQ_N() 4, 3, 2, 1, 0

#define CONCATENATE(x, y) CONCATENATE1(x, y)
#define CONCATENATE1(x, y) x##y

#define FOR_EACH(MACRO, ...)                                                   \
  CONCATENATE(FOR_EACH_, FOR_EACH_NARG_HELPER(__VA_ARGS__))(MACRO, __VA_ARGS__)
#define FOR_EACH_NARG_HELPER(...) FOR_EACH_NARG(__VA_ARGS__)

#define REMOVE_PARENS(...) __VA_ARGS__
#define FOR_EACH_P_INTERMEDIATE(MACRO, ...) FOR_EACH(MACRO, __VA_ARGS__)
#define FOR_EACH_P(MACRO, ARGS_WITH_PARENS)                                    \
  FOR_EACH_P_INTERMEDIATE(MACRO, REMOVE_PARENS ARGS_WITH_PARENS)

#define DECLARE_BACKEND(name) void init_triton_##name(pybind11::module &&m);
#define INIT_BACKEND(name) init_triton_##name(m.def_submodule(#name));

FOR_EACH_P(DECLARE_BACKEND, TRITON_BACKENDS_TUPLE)

#endif // TRITON_HAS_BACKENDS

// ═══════════════════════════════════════════════════════════════════════
// 核心模块初始化
// ═══════════════════════════════════════════════════════════════════════
void init_triton_env_vars(pybind11::module &m);
void init_triton_ir(pybind11::module &&m);
void init_triton_llvm(pybind11::module &&m);
void init_triton_interpreter(pybind11::module &&m);
void init_triton_passes(pybind11::module &&m);
void init_triton_anchor(pybind11::module &&m);

PYBIND11_MODULE(libtriton, m) {
  m.doc() = "Python bindings to the C++ Triton API";
  init_triton_env_vars(m);
  init_triton_ir(m.def_submodule("ir"));
  init_triton_passes(m.def_submodule("passes"));
  init_triton_interpreter(m.def_submodule("interpreter"));
  init_triton_llvm(m.def_submodule("llvm"));
  init_triton_anchor(m.def_submodule("anchor"));
#ifdef TRITON_HAS_BACKENDS
  FOR_EACH_P(INIT_BACKEND, TRITON_BACKENDS_TUPLE)
#endif
}
