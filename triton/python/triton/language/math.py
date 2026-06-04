from triton._no_ttgir import NO_TTGIR as _NO_TTGIR

from . import core
from . import semantic
from functools import wraps
from typing import List

T = core.TypeVar('T')


def _check_dtype(dtypes: List[str]) -> T:
    """
    We're following libdevice's convention to check accepted data types for math functions.
    It is not a good practice to support all data types as accelerators/GPUs don't support
    many float16 and bfloat16 math operations.
    We should let the users know that they are using and invoke explicit cast to convert
    the data type to the supported one.
    """

    def wrapper(fn):

        @wraps(fn)
        def check(*args, **kwargs):
            # concatenate args and kwargs
            all_args = list(args) + list(kwargs.values())
            for arg in [a for a in all_args if isinstance(a, core.tensor)]:
                if arg.type.scalar.name not in dtypes:
                    raise ValueError(f"Expected dtype {dtypes} but got {arg.type.scalar.name}")
            return fn(*args, **kwargs)

        return check

    return wrapper


def _add_math_1arg_docstr(name: str) -> core.Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Computes the element-wise {name} of :code:`x`.

    :param x: the input values
    :type x: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


def _add_math_2arg_docstr(name: str) -> core.Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Computes the element-wise {name} of :code:`x` and :code:`y`.

    :param x: the input values
    :type x: Block
    :param y: the input values
    :type y: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


def _add_math_3arg_docstr(name: str) -> core.Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Computes the element-wise {name} of :code:`x`, :code:`y`, and :code:`z`.

    :param x: the input values
    :type x: Block
    :param y: the input values
    :type y: Block
    :param z: the input values
    :type z: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@core.builtin
@_check_dtype(dtypes=["int32", "int64", "uint32", "uint64"])
@_add_math_2arg_docstr("most significant N bits of the 2N-bit product")
def umulhi(x, y, _builder=None):
    x = core._to_tensor(x, _builder)
    y = core._to_tensor(y, _builder)
    x, y = core.binary_op_type_legalization(x, y, _builder)
    return core.tensor(_builder.create_umulhi(x.handle, y.handle), x.type)


if not _NO_TTGIR:
    @core.builtin
    @_check_dtype(dtypes=["fp32", "fp64", "bf16"])
    @_add_math_1arg_docstr("exponential")
    @core._tensor_member_fn
    def exp(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_exp(x.handle), x.type)
else:
    @core.builtin
    @_check_dtype(dtypes=["fp32", "fp64"])
    @_add_math_1arg_docstr("exponential")
    @core._tensor_member_fn
    def exp(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_exp(x.handle), x.type)


if not _NO_TTGIR:
    @core.builtin
    @_check_dtype(dtypes=["fp32", "fp64", "bf16"])
    @_add_math_1arg_docstr("exponential (base 2)")
    @core._tensor_member_fn
    def exp2(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_exp2(x.handle), x.type)
else:
    @core.builtin
    @_check_dtype(dtypes=["fp32", "fp64"])
    @_add_math_1arg_docstr("exponential (base 2)")
    @core._tensor_member_fn
    def exp2(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_exp2(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("natural logarithm")
@core._tensor_member_fn
def log(x, _builder=None):
    x = core._to_tensor(x, _builder)
    return core.tensor(_builder.create_log(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("logarithm (base 2)")
@core._tensor_member_fn
def log2(x, _builder=None):
    x = core._to_tensor(x, _builder)
    return core.tensor(_builder.create_log2(x.handle), x.type)


if not _NO_TTGIR:
    @core.builtin
    @_check_dtype(dtypes=["fp32", "fp64", "bf16"])
    @_add_math_1arg_docstr("cosine")
    @core._tensor_member_fn
    def cos(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_cos(x.handle), x.type)
else:
    @core.builtin
    @_check_dtype(dtypes=["fp32", "fp64"])
    @_add_math_1arg_docstr("cosine")
    @core._tensor_member_fn
    def cos(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_cos(x.handle), x.type)


if not _NO_TTGIR:
    @core.builtin
    @_check_dtype(dtypes=["fp32", "fp64", "bf16"])
    @_add_math_1arg_docstr("sine")
    @core._tensor_member_fn
    def sin(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_sin(x.handle), x.type)
else:
    @core.builtin
    @_check_dtype(dtypes=["fp32", "fp64"])
    @_add_math_1arg_docstr("sine")
    @core._tensor_member_fn
    def sin(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_sin(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("fast square root")
@core._tensor_member_fn
def sqrt(x, _builder=None):
    x = core._to_tensor(x, _builder)
    return core.tensor(_builder.create_sqrt(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32"])
@_add_math_1arg_docstr("precise square root (rounding to nearest)")
@core._tensor_member_fn
def sqrt_rn(x, _builder=None):
    x = core._to_tensor(x, _builder)
    return core.tensor(_builder.create_precise_sqrt(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("inverse square root")
@core._tensor_member_fn
def rsqrt(x, _builder=None):
    x = core._to_tensor(x, _builder)
    return core.tensor(_builder.create_rsqrt(x.handle), x.type)


@core.builtin
@_add_math_1arg_docstr("absolute value")
@core._tensor_member_fn
def abs(x, _builder=None):
    x = core._to_tensor(x, _builder)
    dtype = x.dtype
    if dtype.is_fp8e4b15():
        mask = core.full(x.shape, 0x7F, core.int8, _builder=_builder)
        return core.tensor(_builder.create_and(x.handle, mask.handle), x.type)
    elif dtype.is_floating():
        return core.tensor(_builder.create_fabs(x.handle), x.type)
    elif dtype.is_int_signed():
        return core.tensor(_builder.create_iabs(x.handle), x.type)
    elif dtype.is_int_unsigned():
        return x  # no-op
    else:
        assert False, f"Unexpected dtype {dtype}"


@core.builtin
@_add_math_2arg_docstr("fast division")
def fdiv(x, y, ieee_rounding=False, _builder=None):
    ieee_rounding = core._constexpr_to_value(ieee_rounding)
    x = core._to_tensor(x, _builder)
    y = core._to_tensor(y, _builder)
    return semantic.fdiv(x, y, ieee_rounding, _builder)


@core.builtin
@_check_dtype(dtypes=["fp32"])
@_add_math_2arg_docstr("precise division (rounding to nearest)")
def div_rn(x, y, _builder=None):
    x = core._to_tensor(x, _builder)
    y = core._to_tensor(y, _builder)
    x, y = core.binary_op_type_legalization(x, y, _builder)
    return core.tensor(_builder.create_precise_divf(x.handle, y.handle), x.type)
if not _NO_TTGIR:


    @core.builtin
    @_check_dtype(dtypes=["fp32"])
    @_add_math_2arg_docstr("precise division (rounding to zero)")
    def div_rz(x, y, _builder=None):
        x = core._to_tensor(x, _builder)
        y = core._to_tensor(y, _builder)
        x, y = core.binary_op_type_legalization(x, y, _builder)
        return core.tensor(_builder.create_precise_divrz(x.handle, y.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("error function")
@core._tensor_member_fn
def erf(x, _builder=None):
    x = core._to_tensor(x, _builder)
    return core.tensor(_builder.create_erf(x.handle), x.type)


if not _NO_TTGIR:
    @core.builtin
    @_check_dtype(dtypes=["fp32", "fp64", "bf16"])
    @_add_math_1arg_docstr("floor")
    @core._tensor_member_fn
    def floor(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_floor(x.handle), x.type)
else:
    @core.builtin
    @_check_dtype(dtypes=["fp32", "fp64"])
    @_add_math_1arg_docstr("floor")
    @core._tensor_member_fn
    def floor(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_floor(x.handle), x.type)


if not _NO_TTGIR:
    @core.builtin
    @_check_dtype(dtypes=["fp32", "fp64", "bf16"])
    @_add_math_1arg_docstr("ceil")
    @core._tensor_member_fn
    def ceil(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_ceil(x.handle), x.type)
else:
    @core.builtin
    @_check_dtype(dtypes=["fp32", "fp64"])
    @_add_math_1arg_docstr("ceil")
    @core._tensor_member_fn
    def ceil(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_ceil(x.handle), x.type)


@core.builtin
@_add_math_3arg_docstr("fused multiply-add")
def fma(x, y, z, _builder=None):
    x = core._to_tensor(x, _builder)
    y = core._to_tensor(y, _builder)
    z = core._to_tensor(z, _builder)
    x, y = core.binary_op_type_legalization(x, y, _builder)
    z, x = core.binary_op_type_legalization(z, x, _builder)
    z, y = core.binary_op_type_legalization(z, y, _builder)
    return core.tensor(_builder.create_fma(x.handle, y.handle, z.handle), x.type)
if not _NO_TTGIR:


    # Generate libdevice op temporarily.
    @core.builtin
    @_check_dtype(dtypes=["fp16", "fp32", "fp64", "int32"])
    @_add_math_2arg_docstr("pow")
    @core._tensor_member_fn
    def pow(x, y, _builder=None):
        x = core._to_tensor(x, _builder)
        y = core._to_tensor(y, _builder)
        dtype = y.dtype

        if dtype.is_int32():
            if y.shape == []:
                mask = core.full(x.shape, 1, core.int32, _builder=_builder)
            else:
                mask = core.broadcast_to(y, x.shape, _builder=_builder)
            return core.tensor(_builder.create_fpowi(x.handle, mask.handle), x.type)
        else:
            x, y = core.binary_op_type_legalization(x, y, _builder)
            return core.tensor(_builder.create_pow(x.handle, y.handle), x.type)


    @core.builtin
    @_check_dtype(dtypes=["fp32", "fp64", "bf16"])
    @_add_math_1arg_docstr("tanh")
    @core._tensor_member_fn
    def tanh(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_tanh(x.handle), x.type)


    @core.builtin
    @_check_dtype(dtypes=["fp32"])
    @_add_math_1arg_docstr("finitef")
    @core._tensor_member_fn
    def finitef(x, _builder=None):
        x = core._to_tensor(x, _builder)
        # Note: type(x.type) --> block_type, and this op returns bool type.
        return core.tensor(_builder.create_isfinite(x.handle), core.block_type(core.dtype("int1"), x.type.get_block_shapes()))


    @core.builtin
    @_check_dtype(dtypes=["fp32"])
    @_add_math_1arg_docstr("isinf")
    @core._tensor_member_fn
    def isinf(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_isinf(x.handle), core.block_type(core.dtype("int1"), x.type.get_block_shapes()))


    @core.builtin
    @_check_dtype(dtypes=["fp32"])
    @_add_math_1arg_docstr("isnan")
    @core._tensor_member_fn
    def isnan(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_isnan(x.handle), core.block_type(core.dtype("int1"), x.type.get_block_shapes()))


    @core.builtin
    @_check_dtype(dtypes=["fp32", "bf16"])
    @_add_math_1arg_docstr("trunc")
    @core._tensor_member_fn
    def trunc(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_math_trunc(x.handle), x.type)


    @core.builtin
    @_check_dtype(dtypes=["fp32", "fp64"])
    @_add_math_2arg_docstr("fmod")
    @core._tensor_member_fn
    def fmod(x, y, _builder=None):
        x = core._to_tensor(x, _builder)
        y = core._to_tensor(y, _builder)
        x, y = core.binary_op_type_legalization(x, y, _builder)
        return core.tensor(_builder.create_fmod(x.handle, y.handle), x.type)


    @core.builtin
    @_check_dtype(dtypes=["fp32", "fp64"])
    @_add_math_1arg_docstr("rint")
    @core._tensor_member_fn
    def rint(x, _builder=None):
        x = core._to_tensor(x, _builder)
        return core.tensor(_builder.create_rint(x.handle), x.type)
