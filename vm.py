"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import types
import typing as tp
import operator
from typing import Any

# sys.setrecursionlimit(int(1e9))

DEBUG = 0
max_lines_debug = 1000
cur_lines_debug = 0


def debug(*args: tp.Any) -> None:
    if not DEBUG:
        return
    global cur_lines_debug
    with open('04.3.HW1/tasks/vm/debug.txt', 'a') as f:
        if cur_lines_debug < max_lines_debug:
            print(*args, file=f)
            cur_lines_debug += 1


CO_VARARGS = 4
CO_VARKEYWORDS = 8
ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
ERR_TOO_MANY_KW_ARGS = 'Too many keyword arguments'
ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
ERR_MISSING_POS_ARGS = 'Missing positional arguments'
ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
ERR_POSONLY_PASSED_AS_KW = 'Positional-only argument passed as keyword argument'


#
#
# def bind_args(code: types.CodeType, *args: Any, **kwargs: Any) -> dict[str, Any]:
#     """Bind values from `args` and `kwargs` to corresponding arguments of `func`
#
#     :param func: function to be inspected
#     :param args: positional arguments to be bound
#     :param kwargs: keyword arguments to be bound
#     :return: `dict[argument_name] = argument_value` if binding was successful,
#              raise TypeError with one of `ERR_*` error descriptions otherwise
#     """
#     flags = code.co_flags
#     varnames = code.co_varnames
#     argcount = code.co_argcount
#     posonlyargcount = code.co_posonlyargcount
#     kwonlyargcount = code.co_kwonlyargcount
#     defaults = code.__defaults__
#     if defaults is None:
#         defaults = ()
#     kwdefaults: dict[str, Any] | None = code.__kwdefaults__
#     if kwdefaults is None:
#         kwdefaults = {}
#
#     if argcount < len(args) and not (flags & CO_VARARGS):
#         raise TypeError(ERR_TOO_MANY_POS_ARGS)
#
#     if posonlyargcount > len(args) and (flags & CO_VARARGS):
#         raise TypeError(ERR_MISSING_POS_ARGS)
#
#     func_args: dict[str, Any] = {}
#     # if len(args) < argcount:
#     #     raise TypeError(ERR_MISSING_POS_ARGS)
#     # if len(args) > argcount:
#     #     raise TypeError(ERR_TOO_MANY_POS_ARGS)
#
#     defaults_iter = 0
#     defaults_value: dict[str, Any] = {}
#     for i in range(argcount - len(defaults), argcount):
#         var = varnames[i]
#         defaults_value[var] = defaults[defaults_iter]
#         defaults_iter += 1
#     for i in range(min(argcount, len(args))):
#         func_args[varnames[i]] = args[i]
#     print('!!!!', func_args, defaults_value)
#     for i in range(argcount):
#         var = varnames[i]
#         if var in func_args:
#             if var in kwargs:
#                 if i < posonlyargcount:
#                     if not (flags & CO_VARKEYWORDS):
#                         raise TypeError(ERR_POSONLY_PASSED_AS_KW)
#                 else:
#                     raise TypeError(ERR_MULT_VALUES_FOR_ARG)
#         elif var in kwargs:
#             if i < posonlyargcount:
#                 raise TypeError(ERR_POSONLY_PASSED_AS_KW)
#             func_args[var] = kwargs[var]
#             kwargs.pop(var)
#         elif var in defaults_value:
#             func_args[var] = defaults_value[var]
#         else:
#             raise TypeError(ERR_MISSING_POS_ARGS)
#
#     for i in range(argcount, argcount + kwonlyargcount):
#         var = varnames[i]
#         if var in kwargs:
#             func_args[var] = kwargs[var]
#             kwargs.pop(var)
#         elif var in kwdefaults:
#             func_args[var] = kwdefaults[var]
#         else:
#             raise TypeError(ERR_MISSING_KWONLY_ARGS)
#
#     if flags & CO_VARKEYWORDS:
#         kwargs_name = varnames[-1]
#         func_args[kwargs_name] = {}
#         for kwarg, value in kwargs.items():
#             func_args[kwargs_name][kwarg] = value
#     elif kwargs:
#         raise TypeError(ERR_POSONLY_PASSED_AS_KW)
#
#     if flags & CO_VARARGS:
#         args_name = varnames[argcount + kwonlyargcount]
#         func_args[args_name] = args[argcount:]
#
#     print(func_args)
#     return func_args


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
        self.instruction_line = 0
        self.kw_name = ()

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
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

    def run(self) -> tp.Any:
        instructions = {instruction.offset: instruction for instruction in dis.get_instructions(self.code)}
        max_offset = max(instructions)
        while self.instruction_line <= max_offset:
            if self.instruction_line not in instructions:
                self.instruction_line += 1
                continue

            instruction = instructions[self.instruction_line]
            opname = instruction.opname.lower() + "_op"
            if opname == 'kw_names_op':
                getattr(self, opname)(instruction.arg)
            else:
                getattr(self, opname)(instruction.argval)

            # debug('===run===')
            # debug(opname, instruction.argval)
            # debug(self.data_stack)
            # debug('---------')
            # debug(instruction)
            # debug('run', self.instruction_line, opname, instruction.argval)
            # debug(self.locals)

            check_jump_instructions = {
                'jump' in opname,
                opname == 'for_iter_op'
            }
            if not any(check_jump_instructions):
                self.instruction_line += 1
        return self.return_value

    def binary_op_op(self, arg: int) -> None:

        binary_op = [
            operator.add,
            operator.and_,
            operator.floordiv,
            operator.lshift,
            operator.matmul,
            operator.mul,
            operator.mod,
            operator.or_,
            operator.pow,
            operator.rshift,
            operator.sub,
            operator.truediv,
            operator.xor,
            operator.iadd,
            operator.iand,
            operator.ifloordiv,
            operator.ilshift,
            operator.imatmul,
            operator.imul,
            operator.imod,
            operator.ior,
            operator.ipow,
            operator.rshift,
            operator.isub,
            operator.itruediv,
            operator.ixor
        ]

        second_arg = self.pop()
        first_arg = self.pop()
        result = binary_op[arg](first_arg, second_arg)
        self.push(result)

    def binary_subscr_op(self, arg: None) -> None:
        i = self.pop()
        seq = self.pop()
        self.push(seq[i])

    def build_const_key_map_op(self, arg: None) -> None:
        keys = self.pop()
        values = self.popn(len(keys))
        dict_ = {key: value for key, value in zip(keys, values)}
        self.push(dict_)

    def build_list_op(self, arg: int) -> None:
        elements = self.popn(arg)
        self.push(elements)

    def build_map_op(self, arg: int) -> None:
        list_ = self.popn(2 * arg)
        debug(list_)
        dict_ = {}
        for i in range(0, len(list_), 2):
            dict_[list_[i]] = list_[i + 1]
        self.push(dict_)

    def build_set_op(self, arg: None) -> None:
        self.push(set())

    def build_slice_op(self, arg: int) -> None:
        if arg == 2:
            a, b = self.popn(2)
            self.push(slice(a, b))
        else:
            a, b, c = self.popn(3)
            self.push((slice(a, b, c)))

    def build_string_op(self, arg: int) -> None:
        strings = self.popn(arg)
        result = ''
        for string in strings:
            result += string
        self.push(result)

    def build_tuple_op(self, arg: int) -> None:
        tuple_ = tuple(self.popn(arg))
        self.push(tuple_)

    def call_function_ex_op(self, arg: int) -> None:
        debug('call_function_ex_op', arg, self.data_stack)
        kwargs = self.pop() if arg & 0x1 else {}
        args = self.pop()
        f = self.pop()
        self.push(f(*args, **kwargs))

    def call_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-CALL
        """

        if arg == 0 and len(self.data_stack) >= 2:
            if hasattr(self.top(), '__iter__') and isinstance(self.data_stack[-2], types.FunctionType):
                tos = self.pop()
                tos1 = self.pop()
                self.push(tos1(tos))
                return

        kw_name_count = len(self.kw_name)
        debug('call_op', arg, self.data_stack, kw_name_count)
        arguments = self.popn(arg)
        kwargs: dict[str, tp.Any] = {}
        for i in range(1, kw_name_count + 1):
            kwargs[self.kw_name[-i]] = arguments[-i]
        if kw_name_count:
            arguments = arguments[:-kw_name_count]
        self.kw_name = ()
        f = self.pop()
        if self.data_stack and self.top() is None:
            self.pop()
        debug('??????', arguments, kwargs)
        self.push(f(*arguments, **kwargs))
        # debug(self.data_stack)

    def compare_op_op(self, op: str) -> None:

        compare_op = {
            '<': operator.le,
            '<=': operator.lt,
            '==': operator.eq,
            '!=': operator.ne,
            '>': operator.ge,
            '>=': operator.gt
        }

        second_arg = self.pop()
        first_arg = self.pop()
        result = compare_op[op](first_arg, second_arg)
        self.push(result)

    def contains_op_op(self, arg: int) -> None:
        tos = self.pop()
        tos1 = self.pop()
        if not arg:
            self.push(tos1 in tos)
        else:
            self.push(tos1 not in tos)

    def copy_op(self, i: int) -> None:
        obj = self.data_stack[-i]
        if hasattr(obj, 'copy'):
            self.push(obj.copy())
        else:
            self.push(obj)

    def delete_attr_op(self, name: str) -> None:
        tos = self.pop()
        delattr(tos, name)

    def delete_fast_op(self, name: str) -> None:
        del self.locals[name]

    def delete_global_op(self, name: str) -> None:
        del self.globals[name]

    def delete_subscr_op(self, arg: None) -> None:
        tos = self.pop()
        tos1 = self.pop()
        del tos1[tos]

    def dict_merge_op(self, arg: int) -> None:
        dict_ = self.pop()
        for key, value in dict_.items():
            if key not in self.data_stack[-arg]:
                self.data_stack[-arg][key] = value
            else:
                raise KeyError

    def dict_update_op(self, arg: int) -> None:
        dict_ = self.pop()
        dict.update(self.data_stack[-arg], dict_)

    def for_iter_op(self, arg: int) -> None:
        try:
            value = next(self.top())
            self.push(value)
            self.instruction_line += 1
        except StopIteration:
            self.pop()
            self.instruction_line = arg

    def format_value_op(self, arg: tuple[tp.Any, ...]) -> None:
        if arg[1]:
            frt_spec = self.pop()
        else:
            frt_spec = ''
        value = self.pop()
        if arg[0] is not None:
            value = arg[0](value)
        debug(arg, value)
        # if (flags & 0x03) == 0x01:
        #     value = str(value)
        # if (flags & 0x03) == 0x02:
        #     value = repr(value)
        # if (flags & 0x03) == 0x02:
        #     value = ascii(value)
        self.push(format(value, frt_spec))

    def get_iter_op(self, arg: None) -> None:
        iterable = self.pop()
        self.push(iter(iterable))

    def raise_varargs_op(self, arg: int) -> None:
        # if arg == 0:
        #     raise
        # if arg == 1:
        #     raise self.pop()
        # if arg == 2:
        #     tos = self.pop()
        #     tos1 = self.pop()
        #     raise tos1 from tos
        pass

    def resume_op(self, arg: int) -> tp.Any:
        pass

    def push_null_op(self, arg: int) -> tp.Any:
        self.push(None)

    def precall_op(self, arg: int) -> tp.Any:
        pass

    def is_op_op(self, arg: int) -> tp.Any:
        tos = self.pop()
        tos1 = self.pop()
        if not arg:
            self.push(tos1 is tos)
        else:
            self.push(tos1 is not tos)

    def jump_backward_op(self, arg: int) -> None:
        self.instruction_line = arg

    def jump_forward_op(self, arg: int) -> None:
        self.instruction_line = arg

    def jump_if_false_or_pop_op(self, arg: int) -> None:
        if not self.top():
            self.instruction_line = arg
        else:
            self.instruction_line += arg
            self.pop()

    def jump_if_true_or_pop_op(self, arg: int) -> None:
        if self.top():
            self.instruction_line = arg
        else:
            self.instruction_line += 1
            self.pop()

    def kw_names_op(self, arg: int) -> None:
        self.kw_name = self.code.co_consts[arg]

    def list_append_op(self, i: int) -> None:
        tos = self.pop()
        list.append(self.data_stack[-i], tos)

    def list_extend_op(self, arg: int) -> None:
        debug(arg, self.data_stack)
        list_ = self.pop()
        list.extend(self.data_stack[-arg], list_)

    def list_to_tuple_op(self, arg: None) -> None:
        list_ = self.pop()
        self.push(tuple(list_))

    def load_assertion_error_op(self, arg: tp.Any) -> None:
        pass

    def load_attr_op(self, name: str) -> None:
        tos = self.pop()
        self.push(getattr(tos, name))

    def load_fast_op(self, arg: str) -> None:
        if arg in self.locals:
            self.push(self.locals[arg])
        else:
            raise UnboundLocalError

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

    def load_global_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_GLOBAL
        """
        if arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError

    def load_method_op(self, arg: str) -> None:
        self_ = self.pop()
        if hasattr(self_, arg):
            self.push(getattr(self_, arg))
        else:
            self.push(None)

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(arg)

    def return_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-RETURN_VALUE
        """
        self.return_value = self.pop()

    def make_function_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-MAKE_FUNCTION
        """
        debug('make_function_op')
        debug(arg, self.data_stack)

        code = self.pop()  # the code associated with the function (at TOS1)

        flags = code.co_flags
        varnames = code.co_varnames
        argcount = code.co_argcount
        posonlyargcount = code.co_posonlyargcount
        kwonlyargcount = code.co_kwonlyargcount
        # defaults = self.pop() if arg else []
        if arg & 0x02:
            kwdefaults = self.pop()
        else:
            kwdefaults = {}

        if arg:
            defaults = self.pop()
            debug('!!!!?!?!?', defaults, arg)
        else:
            defaults = []
        debug("kwargs = ", kwonlyargcount)

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:

            debug('!!!!!!!!!', kwargs, arg)

            func_args: dict[str, tp.Any] = {}

            defaults_iter = 0
            defaults_value: dict[str, Any] = {}
            for i in range(argcount - len(defaults), argcount):
                var = varnames[i]
                debug('??', defaults, defaults_iter)
                defaults_value[var] = defaults[defaults_iter]
                defaults_iter += 1
            for i in range(min(argcount, len(args))):
                func_args[varnames[i]] = args[i]
            for i in range(argcount):
                var = varnames[i]
                if var in func_args:
                    if var in kwargs:
                        if i < posonlyargcount:
                            if not (flags & CO_VARKEYWORDS):
                                raise TypeError(ERR_POSONLY_PASSED_AS_KW)
                        else:
                            raise TypeError(ERR_MULT_VALUES_FOR_ARG)
                elif var in kwargs:
                    if i < posonlyargcount:
                        raise TypeError(ERR_POSONLY_PASSED_AS_KW)
                    func_args[var] = kwargs[var]
                    kwargs.pop(var)
                elif var in defaults_value:
                    func_args[var] = defaults_value[var]
                else:
                    raise TypeError(ERR_MISSING_POS_ARGS)

            for i in range(argcount, argcount + kwonlyargcount):
                var = varnames[i]
                if var in kwargs:
                    func_args[var] = kwargs[var]
                    kwargs.pop(var)
                elif var in kwdefaults:
                    func_args[var] = kwdefaults[var]
                else:
                    raise TypeError(ERR_MISSING_KWONLY_ARGS)

            if flags & CO_VARKEYWORDS:
                kwargs_name = varnames[-1]
                func_args[kwargs_name] = {}
                for kwarg, value in kwargs.items():
                    func_args[kwargs_name][kwarg] = value
            elif kwargs:
                raise TypeError(ERR_POSONLY_PASSED_AS_KW)

            if flags & CO_VARARGS:
                args_name = varnames[argcount + kwonlyargcount]
                func_args[args_name] = args[argcount:]
            debug('func_args =', func_args)
            f_locals = dict(self.locals)
            f_locals.update(func_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)  # Run code in prepared environment
            return frame.run()

        self.push(f)

    def map_add_op(self, i: int) -> None:
        tos = self.pop()
        tos1 = self.pop()
        dict.__setitem__(self.data_stack[-i], tos1, tos)

    def nop_op(self, arg: int) -> None:
        pass

    def pop_jump_forward_if_false_op(self, arg: int) -> None:
        if not self.pop():
            self.instruction_line = arg
        else:
            self.instruction_line += 1

    def pop_jump_forward_if_none_op(self, arg: int) -> None:
        if self.pop() is None:
            self.instruction_line = arg
        else:
            self.instruction_line += 1

    def pop_jump_forward_if_not_none_op(self, arg: int) -> None:
        if self.pop() is not None:
            self.instruction_line = arg
        else:
            self.instruction_line += 1

    def pop_jump_forward_if_true_op(self, arg: int) -> None:
        if self.pop():
            self.instruction_line = arg
        else:
            self.instruction_line += 1

    def pop_top_op(self, arg: None) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-POP_TOP
        """
        self.pop()

    def set_add_op(self, i: int) -> None:
        tos = self.pop()
        set.add(self.data_stack[-i], tos)

    def set_update_op(self, arg: int) -> None:
        set_ = self.pop()
        set.update(self.data_stack[-arg], set_)

    def setup_annotations_op(self, arg: None) -> None:
        if '__annotations__' not in self.locals:
            self.locals['__annotations__'] = {}

    def store_attr_op(self, name: str) -> None:
        tos = self.pop()
        tos1 = self.pop()
        setattr(tos, name, tos1)

    def store_fast_op(self, arg: str) -> None:
        const = self.pop()
        self.locals[arg] = const

    def store_global_op(self, arg: str) -> None:
        const = self.pop()
        self.globals[arg] = const

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-STORE_NAME
        """
        const = self.pop()
        self.locals[arg] = const

    def store_subscr_op(self, arg:  None) -> None:
        tos = self.pop()
        tos1 = self.pop()
        tos2 = self.pop()
        tos1[tos] = tos2

    def swap_op(self, i: int) -> None:
        self.data_stack[-i], self.data_stack[-1] = \
            self.data_stack[-1], self.data_stack[-i]

    def unary_invert_op(self, arg: None) -> None:
        self.push(~self.pop())

    def unary_negative_op(self, arg: None) -> None:
        self.push(-self.pop())

    def unary_not_op(self, arg: None) -> None:
        self.push(not self.pop())

    def unary_positive_op(self, arg: None) -> None:
        pass

    def unpack_sequence_op(self, arg: int) -> None:
        values = self.pop()
        debug('unpack_sequence_op', values)
        self.push(*values)


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()
