#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Signals.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define FOR_EACH_1(MACRO, X) MACRO(X)
#define FOR_EACH_2(MACRO, X, ...) MACRO(X) FOR_EACH_1(MACRO, __VA_ARGS__)
#define FOR_EACH_3(MACRO, X, ...) MACRO(X) FOR_EACH_2(MACRO, __VA_ARGS__)
#define FOR_EACH_4(MACRO, X, ...) MACRO(X) FOR_EACH_3(MACRO, __VA_ARGS__)
#define FOR_EACH_5(MACRO, X, ...) MACRO(X) FOR_EACH_4(MACRO, __VA_ARGS__)
#define FOR_EACH_6(MACRO, X, ...) MACRO(X) FOR_EACH_5(MACRO, __VA_ARGS__)

#define FOR_EACH_NARG(...) FOR_EACH_NARG_(__VA_ARGS__, FOR_EACH_RSEQ_N())
#define FOR_EACH_NARG_(...) FOR_EACH_ARG_N(__VA_ARGS__)
#define FOR_EACH_ARG_N(_1, _2, _3, _4, _5, _6, N, ...) N
#define FOR_EACH_RSEQ_N() 6, 5, 4, 3, 2, 1, 0

#define CONCATENATE(x, y) CONCATENATE1(x, y)
#define CONCATENATE1(x, y) x##y

#define FOR_EACH(MACRO, ...)                                                   \
  CONCATENATE(FOR_EACH_, FOR_EACH_NARG_HELPER(__VA_ARGS__))(MACRO, __VA_ARGS__)
#define FOR_EACH_NARG_HELPER(...) FOR_EACH_NARG(__VA_ARGS__)

// New macro to remove parentheses
#define REMOVE_PARENS(...) __VA_ARGS__

// Intermediate macro to ensure correct expansion
#define FOR_EACH_P_INTERMEDIATE(MACRO, ...) FOR_EACH(MACRO, __VA_ARGS__)

// Modified FOR_EACH to handle parentheses
#define FOR_EACH_P(MACRO, ARGS_WITH_PARENS)                                    \
  FOR_EACH_P_INTERMEDIATE(MACRO, REMOVE_PARENS ARGS_WITH_PARENS)

#define DECLARE_BACKEND(name) void init_triton_##name(pybind11::module &&m);

#define INIT_BACKEND(name) init_triton_##name(m.def_submodule(#name));

void init_triton_env_vars(pybind11::module &m);
void init_triton_ir(pybind11::module &&m);
void init_triton_llvm(pybind11::module &&m);
void init_triton_interpreter(pybind11::module &&m);
void init_triton_passes(pybind11::module &&m);
void init_triton_stacktrace_hook(pybind11::module &m);
// triton-anchor 扩展模块（注册 triton-shared 方言和通用 Pass）
void init_triton_anchor(pybind11::module &&m);
// triton-anchor 不编译任何后端到 libtriton.so 中 —— 后端通过独立包加载。
// 当 TRITON_BACKENDS_TUPLE=() 时，FOR_EACH_P 的空展开会生成
// init_triton_(pybind11::module&&) 这样的无效声明，因此不调用。

PYBIND11_MODULE(libtriton, m) {
  m.doc() = "Python bindings to the C++ Triton API";
  init_triton_stacktrace_hook(m);
  init_triton_env_vars(m);
  init_triton_ir(m.def_submodule("ir"));
  init_triton_passes(m.def_submodule("passes"));
  init_triton_interpreter(m.def_submodule("interpreter"));
  init_triton_llvm(m.def_submodule("llvm"));
  // 注册 triton-anchor 扩展（triton-shared 方言 + 通用 Pass）
  init_triton_anchor(m.def_submodule("anchor"));
  // triton-anchor 不内置任何硬件后端，后端初始化由外部插件完成。
  // FOR_EACH_P(INIT_BACKEND, TRITON_BACKENDS_TUPLE) 已移除。
}
