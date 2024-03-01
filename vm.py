"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import operator
import types
import typing as tp
from typing import Any


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.11/Include/frameobject.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """
    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value = None
        self.kwargs: dict[str, tp.Any] = {}

    def top(self) -> tp.Any:
        if len(self.data_stack) < 1:
            raise Exception
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        if len(self.data_stack) < 1:
            raise Exception
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    # @functools.lru_cache(256)
    def run(self) -> tp.Any:
        instruction = list(dis.get_instructions(self.code))
        ind = 0
        while True:
            f_tmp = getattr(self, instruction[ind].opname.lower() + "_op")
            opname = instruction[ind].opname
            boolean_value = (opname == 'KW_NAMES') | (opname == 'LOAD_GLOBAL')
            boolean_value = boolean_value | (opname == 'LOAD_METHOD')
            # offset = f_tmp(instruction[ind].argval, instruction[ind].arg)
            if boolean_value:
                offset = f_tmp(instruction[ind].argval, instruction[ind].arg)
            else:
                offset = f_tmp(instruction[ind].argval)
            if offset is not None:
                while offset != instruction[ind].offset:
                    if instruction[ind].offset < offset:
                        ind += 1
                    else:
                        ind -= 1
                continue
            ind += 1
            if ind >= len(instruction):
                break
        return self.return_value

    def resume_op(self, arg: int) -> tp.Any:
        pass

    def push_null_op(self, arg: int) -> tp.Any:
        self.push(None)

    def precall_op(self, arg: int) -> tp.Any:
        pass

    # @functools.lru_cache(256)
    def call_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-CALL
        """
        args = self.popn(arg + 2)

        kwargs = {}

        for k, v in zip(list(self.kwargs)[::-1], args[::-1]):
            kwargs[k] = v
        self.kwargs = {}
        args_len = len(args) - len(self.kwargs)
        args = args[:args_len]
        func_res = args[0]
        ind = 2
        first_none = func_res is None
        if first_none:
            func_res = args[1]
        if args_len != arg + 2:
            ind = 1
        self.push(func_res(*args[ind::], **kwargs))

    # @functools.lru_cache(256)
    def call_function_ex_op(self, flag: int) -> None:
        kwargs = {}
        if flag % 2 == 1:
            kwargs = self.pop()
        args = self.pop()
        func = self.pop()
        self.push(func(*args, **kwargs))

    def raise_varargs_op(self, arg: int) -> None:
        if arg == 0:  # 0: raise (re-raise previous exception)
            raise
        elif arg == 1:
            # 1: raise TOS (raise exception instance or type at TOS)
            tos = self.top()
            self.current_exception = tos
            raise tos
        elif arg == 2:
            # raise TOS1 from TOS
            # (raise exception instance
            # or type at TOS1 with __cause__ set to TOS)
            tos1, tos = self.popn(2)
            cause_kwarg = tos1['__cause__']
            self.current_exception = tos
            raise cause_kwarg

    def push_exc_info_op(self, arg: None) -> None:
        tos = self.pop()
        self.push(self.current_exception)
        self.push(tos)

    def check_exc_match_op(self, arg: None) -> None:
        tos = self.pop()
        self.push(tos == self.top())

    def pop_except_op(self, arg: None) -> None:
        self.pop()

    def kw_names_op(self, arg: None, key: int) -> None:
        self.kwargs = self.code.co_consts[key]

    def make_cell_op(self, arg: int) -> None:
        pass

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_NAME
        """
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError

    def load_global_op(self, arg: str, name_ind: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_GLOBAL
        """
        if (name_ind % 2) == 1:
            self.push(None)
        # TODO: parse all scopes
        # self.push(self.globals[arg])
        if arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError
        # self.push(self.builtins[arg])

    def load_method_op(self, name: str, name_ind: int) -> None:
        tos_obj = self.pop()
        method_name = getattr(tos_obj, name, '')
        if method_name:
            self.push(method_name)
            self.push(tos_obj)
        else:
            self.push(None)
            self.push(tos_obj)

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(arg)

    def load_fast_op(self, arg: tp.Any) -> None:
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])

    def load_attr_op(self, arg: Any) -> None:
        tmp = getattr(self.pop(), arg)
        self.push(tmp)

    def load_assertion_error_op(self, arg: int) -> None:
        # Pushes AssertionError onto the stack. Used by the assert statement.
        self.push(AssertionError)

    def reraise_op(self, arg: int) -> None:
        raise self.top()

    def load_build_class_op(self, arg: int) -> None:
        self.push(self.builtins['__build_class__'])

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-STORE_NAME
        """
        const = self.pop()
        self.locals[arg] = const

    def store_global_op(self, arg: str) -> None:
        const = self.pop()
        self.globals[arg] = const
        return None

    def store_fast_op(self, arg: str) -> None:
        const = self.pop()
        self.locals[arg] = const

    def store_subscr_op(self, arg: int) -> None:
        # Implements TOS1[TOS] = TOS2.
        tos2, tos1, tos = self.popn(3)
        tos1[tos] = tos2

    def store_attr_op(self, arg: Any) -> None:
        # TOS.name = TOS1, where namei is the index of name in co_names.
        tos1, tos = self.popn(2)
        setattr(tos, arg, tos1)
        self.push(tos1)
        self.push(tos)

    # def delete_name_op(self, arg: Any) -> None:
    #     del self.code.co_names[arg]

    def delete_subscr_op(self, arg: int) -> None:
        tos1, tos = self.popn(2)
        del tos1[tos]  # Implements del TOS1[TOS]

    def delete_fast_op(self, arg: Any) -> None:
        self.locals.pop(arg)

    def delete_global_op(self, arg: Any) -> None:
        self.globals.pop(arg)

    def delete_attr_op(self, namei: Any) -> None:
        # del TOS.name, using namei as index into co_names.
        delattr(self.top(), namei)

    def import_name_op(self, name_ind: Any) -> None:
        # TOS and TOS1 are popped and provide the fromlist and level argument
        level, fromlist = self.popn(2)
        lst = [name_ind, self.globals, self.locals]
        import_ = __import__(lst[0], lst[1], lst[2], fromlist, level)
        self.push(import_)

    def import_from_op(self, arg: str) -> None:
        const = getattr(self.top(), arg)
        self.push(const)

    def import_star_op(self, arg: int) -> None:
        tos = self.pop()
        for symbol in dir(tos):
            if symbol[0] != '_':
                self.locals[symbol] = getattr(tos, symbol)

    def get_len_op(self) -> None:
        # Push len(TOS) onto the stack.
        tos = self.pop()
        self.push(len(tos))

    def return_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-RETURN_VALUE
        """
        self.return_value = self.pop()

    def pop_top_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-POP_TOP
        """
        self.pop()

    # @functools.lru_cache(256)
    def make_function_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-MAKE_FUNCTION
        """
        code = self.pop()  # the code associated with the function (at TOS1)
        pos_tuple = ()
        keyword_only_dict = {}
        if arg == 0x01:
            pos_tuple = self.pop()
        elif arg == 0x02:
            keyword_only_dict = self.pop()
        elif arg == 0x03:
            keyword_only_dict = self.pop()
            pos_tuple = self.pop()

        # TODO: use arg to parse function defaults

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            # parsed_args: dict[str, tp.Any] = {}
            kwargs.update(keyword_only_dict)
            lst = [pos_tuple, keyword_only_dict, code]
            parsed_args: dict[str, tp.Any] = bind_args(lst, *args, **kwargs)
            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)
            # Run code in prepared environment
            return frame.run()

        self.push(f)

    def binary_subscr_op(self, arg: Any) -> None:
        # Implements TOS = TOS1[TOS].
        tos1, tos = self.popn(2)
        self.push(tos1[tos])

    def binary_op_op(self, op: Any) -> None:
        x, y = self.popn(2)
        if op in BINARY_OPS:
            self.push(BINARY_OPS[op](x, y))
        elif op in INPLACE_OPS:
            self.push(INPLACE_OPS[op](x, y))

    def compare_op_op(self, op: Any) -> None:
        x, y = self.popn(2)
        if op in COMPARE_OPS:
            self.push(COMPARE_OPS[op](x, y))
        else:
            raise Exception

    def contains_op_op(self, arg: Any) -> None:
        # Performs in comparison, or not in if invert is 1.
        # print(self.data_stack)
        x, y = self.popn(2)
        # if len(self.data_stack) > 0:
        #     print(self.pop())
        # print(x, y)
        boolean_value = x in y
        if arg == 1:
            boolean_value = x not in y
        self.push(boolean_value)
        # print(self.data_stack)

    def is_op_op(self, arg: Any) -> None:
        # Performs is comparison, or is not if invert is 1.
        x, y = self.popn(2)
        boolean_value = x is y
        if arg == 1:
            boolean_value = x is not y
        self.push(boolean_value)

    def nop_op(self, arg: Any) -> None:
        pass

    def unary_positive_op(self, arg: int) -> None:
        v = +self.pop()
        self.push(v)

    def unary_negative_op(self, arg: int) -> None:
        v = -self.pop()
        self.push(v)

    def unary_not_op(self, arg: Any) -> None:
        v = not self.pop()
        self.push(v)

    def unary_invert_op(self, arg: Any) -> None:
        v = ~self.pop()
        self.push(v)

    def build_slice_op(self, arg: Any) -> None:
        x, y = self.popn(2)
        if arg == 3:
            tos = self.pop()
            self.push(slice(tos, x, y))
        else:
            self.push(slice(x, y))

    def build_list_op(self, arg: int) -> None:
        n_args = self.popn(arg)
        self.push(list(n_args))

    def build_set_op(self, arg: int) -> None:
        n_args = self.popn(arg)
        self.push(set(n_args))

    def build_tuple_op(self, arg: int) -> None:
        n_args = self.popn(arg)
        self.push(tuple(n_args))

    def build_string_op(self, arg: int) -> None:
        n_args = self.popn(arg)
        self.push(''.join(n_args))

    def build_map_op(self, arg: Any) -> None:
        items = self.popn(arg * 2)  # Pops 2 * count items
        # ..., TOS3: TOS2, TOS1: TOS
        map_to_push = {items[2 * i]: items[2 * i + 1] for i in range(0, arg)}
        self.push(map_to_push)

    def build_const_key_map_op(self, arg: Any) -> None:
        # Pops the top element on the stack which contains a tuple of keys
        tmp = self.pop()
        # then starting from TOS1,
        # pops count values to form values in the built dct.
        values = self.popn(arg)
        self.push(dict(zip(tmp, values)))

    def list_extend_op(self, arg: int) -> None:
        # Calls list.extend(TOS1[-i], TOS). Used to build lists.
        tos = self.pop()
        list.extend(self.data_stack[-arg], tos)

    def list_append_op(self, arg: Any) -> None:
        tos1, tos = self.popn(2)
        list.append(tos1[-arg], tos)
        # self.push(list.append(tos1[-arg], tos))

    def set_update_op(self, arg: int) -> None:
        # Calls set.update(TOS1[-i], TOS). Used to build sets.
        tos = self.pop()
        set.update(self.data_stack[-arg], tos)
        # tos1, tos = self.popn(2)
        # set.update(tos1[-arg], tos)

    def set_add_op(self, arg: int) -> None:
        # Calls set.update(TOS1[-i], TOS). Used to build sets.
        # tos1, tos = self.popn(2)
        # set.add(tos1[-arg], tos)
        tos = self.pop()
        set.add(self.data_stack[-arg], tos)

    def dict_update_op(self, arg: int) -> None:
        # Calls dict.update(TOS1[-i], TOS). Used to build dicts.
        tos = self.pop()
        dict.update(self.data_stack[-arg], tos)

    def dict_merge_op(self, arg: Any) -> None:
        tos = self.pop()
        dict.update(self.data_stack[-arg], tos)

    def list_to_tuple_op(self, arg: Any) -> None:
        self.push(tuple(self.pop()))

    def copy_op(self, arg: Any) -> None:
        const = self.data_stack[-arg]
        self.push(const)

    def swap_op(self, arg: Any) -> None:
        # Swap TOS with the item at position i.
        tmp = self.data_stack[-arg]
        self.data_stack[-arg] = self.data_stack[-1]
        self.data_stack[-1] = tmp
        # self.data_stack[-arg], self.data_stack[-1]
        # = self.data_stack[-1], self.data_stack[-arg]

    def unpack_sequence_op(self, arg: Any) -> None:
        const = self.pop()
        for i in range(len(const) - 1, -1, -1):
            self.push(const[i])

    def pop_jump_forward_if_true_op(self, name_ind: int) -> None | int:
        # if self.pop():
        #     return name_ind
        # return None
        tos = self.pop()
        if tos:
            return name_ind
        return None

    def pop_jump_forward_if_false_op(self, name_ind: int) -> None | int:
        # if not self.pop():
        #     return name_ind
        # return None
        tos = self.pop()
        if not tos:
            return name_ind
        return None

    def jump_if_true_or_pop_op(self, name_ind: int) -> None | int:
        if self.top():
            return name_ind
        self.pop()
        return None

    def jump_if_false_or_pop_op(self, name_ind: int) -> None | int:
        if not self.top():
            return name_ind
        self.pop()
        return None

    def pop_jump_forward_if_none_op(self, name_ind: int) -> None | int:
        tos = self.pop()
        if tos is None:
            return name_ind
        return None

    def pop_jump_backward_if_none_op(self, name_ind: int) -> None | int:
        tos = self.pop()
        if tos is None:
            return name_ind
        return None

    def pop_jump_forward_if_not_none_op(self, name_ind: int) -> None | int:
        tos = self.pop()
        if tos is not None:
            return name_ind
        return None

    def pop_jump_backward_if_not_none_op(self, name_ind: int) -> None | int:
        tos = self.pop()
        if tos is not None:
            return name_ind
        return None

    def jump_backward_no_interrupt(self, delta: int) -> int:
        return delta

    def jump_forward_op(self, arg: int) -> int:
        return arg

    def jump_backward_op(self, arg: int) -> int:
        return arg

    def get_iter_op(self, arg: Any) -> None:
        # TOS = iter(TOS)
        tos = iter(self.pop())
        self.push(tos)

    def for_iter_op(self, delta: int) -> None | int:
        tos_iterator = self.top()  # TOS is an iterator.
        try:
            # If this yields a new value, push it on the stack
            new_value = next(tos_iterator)
            self.push(new_value)
            return None
        except StopIteration:
            # TOS is popped, and the byte code counter is incremented by delta.
            self.pop()
            return delta

    def format_value_op(self, flags: Any) -> None:
        # assert arg[0] == str or arg[0] is None or arg[0] == repr

        if (flags[0] & 0x03) == 0x00:
            self.push(self.pop())
        elif (flags[0] & 0x03) == 0x01:
            self.push(str(self.pop()))
        elif (flags[0] & 0x03) == 0x02:
            self.push(repr(self.pop()))
        elif (flags[0] & 0x03) == 0x03:
            self.push(ascii(self.pop()))
        elif (flags[0] & 0x04) == 0x04:
            # if len(self.data_stack) > 0:
            #     fmt_spec = self.pop()
            pass

    def setup_annotations_op(self, arg: int) -> None:
        if '__annotations__' not in self.locals:
            self.locals['__annotations__'] = {}

    def match_keys(self) -> None:
        # tos1, tos = self.popn(2)
        pass


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        g_c: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], g_c, g_c)
        return frame.run()


BINARY_OPS = {
    8: operator.pow,
    5: operator.mul,
    2: operator.floordiv,
    11: operator.truediv,
    6: operator.mod,
    0: operator.add,
    10: operator.sub,
    3: operator.lshift,
    9: operator.rshift,
    1: operator.and_,
    12: operator.xor,
    7: operator.or_,
}
COMPARE_OPS = {
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '>=': operator.ge,
}
INPLACE_OPS = {
    21: operator.ipow,
    18: operator.imul,
    15: operator.ifloordiv,
    24: operator.itruediv,
    19: operator.imod,
    13: operator.iadd,
    23: operator.isub,
    16: operator.ilshift,
    22: operator.irshift,
    14: operator.iand,
    25: operator.ixor,
    20: operator.ior,
}

CO_VARARGS = 4
CO_VARKEYWORDS = 8

ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
ERR_TOO_MANY_KW_ARGS = 'Too many keyword arguments'
ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
ERR_MISSING_POS_ARGS = 'Missing positional arguments'
ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
s = 'Positional-only argument passed as keyword argument'
ERR_POSONLY_PASSED_AS_KW = s


def bind_args(lst: list[Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Bind values from `args` and `kwargs` to corresponding arguments of `func
    :param func: function to be inspected
    :param args: positional arguments to be bound
    :param kwargs: keyword arguments to be bound
    :return: `dict[argument_name] = argument_value` if binding was successful,
             raise TypeError with one of `ERR_*` error descriptions otherwise
    """
    pos_tuple, keyword_only_dict, func = lst[0], lst[1], lst[2]
    total_dict = {}

    var_names = func.co_varnames
    co_flags = func.co_flags
    co_posonlyargcount = func.co_posonlyargcount
    co_argcount = func.co_argcount
    co_kwonlyargcount = func.co_kwonlyargcount
    default_values = pos_tuple or ()
    default_varnames = var_names[co_argcount - len(default_values):co_argcount]
    args_v = None
    kwargs_v = None

    if keyword_only_dict is not None:
        for k, v in keyword_only_dict.items():
            total_dict[k] = v

    for i in range(0, len(default_varnames)):
        total_dict[default_varnames[i]] = default_values[i]
    arg_v_ind = None
    for i, v in enumerate(var_names[co_argcount:]):
        if (v in kwargs) | (v in total_dict):
            continue
        if ((co_flags & CO_VARARGS) == CO_VARARGS) & (args_v is None):
            args_v = v
            arg_v_ind = i
            continue
        boolean_value = co_flags & CO_VARKEYWORDS
        if (boolean_value == CO_VARKEYWORDS) & (kwargs_v is None):
            kwargs_v = v
            continue
    res_found = {}

    if kwargs_v in var_names:
        total_dict[kwargs_v] = {}

    for i in range(0, co_posonlyargcount):
        if (var_names[i] in kwargs) & (kwargs_v is None):
            raise TypeError(ERR_POSONLY_PASSED_AS_KW)
        elif (var_names[i] in kwargs) & (kwargs_v is not None):
            n = var_names[i]
            total_dict[kwargs_v][n] = kwargs[n]

    for k, v in kwargs.items():
        if kwargs_v is not None:
            if k in total_dict[kwargs_v]:
                continue
        res_found[k] = v
        if k in var_names:
            total_dict[k] = v
        elif kwargs_v in var_names:
            total_dict[kwargs_v][k] = v

    possible_args = []
    pos_cnt = 0
    is_default = False
    if arg_v_ind is not None:
        for i in range(0, arg_v_ind):
            if var_names[i] in res_found:
                raise TypeError(ERR_MISSING_POS_ARGS)
                # raise TypeError
    if arg_v_ind is not None:
        for i in range(arg_v_ind, arg_v_ind + co_kwonlyargcount):
            if var_names[i] not in total_dict:
                raise TypeError(ERR_MISSING_KWONLY_ARGS)
                # raise TypeError
    for i in range(0, len(args)):
        if is_default is False:
            if i >= len(var_names):
                # raise TypeError
                raise TypeError(ERR_TOO_MANY_POS_ARGS)
            if (args_v == var_names[i]) | (kwargs_v == var_names[i]):
                is_default = True
                possible_args.append(args[i])
                continue
        if is_default:
            possible_args.append(args[i])
            continue
        if var_names[i] in res_found:
            if i >= co_argcount:
                is_default = True
                possible_args.append(args[i])
                continue
            # raise TypeError
            raise TypeError(ERR_MULT_VALUES_FOR_ARG)
        total_dict[var_names[i]] = args[i]
        pos_cnt += 1
        res_found[var_names[i]] = args[i]

    if args_v in var_names:
        total_dict[args_v] = tuple(possible_args)

    if pos_cnt > co_argcount:
        # raise TypeError
        raise TypeError(ERR_TOO_MANY_POS_ARGS)

    if len(total_dict) < co_argcount:
        raise TypeError(ERR_MISSING_POS_ARGS)
        # raise TypeError
    if len(total_dict) < co_kwonlyargcount:
        raise TypeError(ERR_MISSING_KWONLY_ARGS)
        # raise TypeError
    return total_dict
