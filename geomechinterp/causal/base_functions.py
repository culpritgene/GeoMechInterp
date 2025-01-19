from typing import Callable


ALL_BINARY_FEATURES = ["position_parity", "ab", "case", "12", "+-", "><", "?!", "]["]


def _space_check(fn: Callable):
    def wrapper(s: str):
        if " " in s:
            raise ValueError("space_check: string must not contain spaces")
        return fn(s)

    return wrapper


def position_parity_check(s: str, parity: int) -> bool:
    # a bit ugly skipping s, but it works
    return parity % 2 == 0


@_space_check
def ab_check(s: str) -> bool:
    return True if "a" in s else False


@_space_check
def case_check(s: str) -> bool:
    return True if s.isupper() else False


@_space_check
def f12_check(s: str) -> bool:
    return True if "1" in s else False


@_space_check
def plus_minus_check(s: str) -> bool:
    return True if "+" in s else False


@_space_check
def qm_check(s: str) -> bool:
    return True if "?" in s else False


@_space_check
def bracket_check(s: str) -> bool:
    return True if "]" in s else False


def ab_f(s: str, c: int) -> str:
    return s + "a" if c == 1 else s + "b"


def case_f(s: str, c: int) -> str:
    return s.upper() if c == 1 else s.lower()


def f12_f(s: str, c: int) -> str:
    return s + "1" if c == 1 else s + "2"


def plus_minus_f(s: str, c: int) -> str:
    return s + "+" if c == 1 else s + "-"


def qm_f(s: str, c: int) -> str:
    return s + "?" if c == 1 else s + "!"


def bracket_f(s: str, c: int) -> str:
    return s + "]" if c == 1 else s + "["


ALL_BINARY_GENERATORS = {
    "position_parity": position_parity_check,  # no control!
    "ab": ab_f,
    "case": case_f,
    "12": f12_f,
    "+-": plus_minus_f,
    "><": qm_f,
    "][": bracket_f,
}

FUNCTION_NAME_TO_FEATURE = {v.__name__: k for k, v in ALL_BINARY_GENERATORS.items()}

ALL_BINARY_GENERATOR_FUNC_NAMES = {
    "position_parity_check": position_parity_check,  # no control!
    "ab_f": ab_f,
    "case_f": case_f,
    "f12_f": f12_f,
    "plus_minus_f": plus_minus_f,
    "qm_f": qm_f,
    "bracket_f": bracket_f,
}


ALL_BINARY_CHECKS = {
    "position_parity": position_parity_check,
    "ab": ab_check,
    "case": case_check,
    "12": f12_check,
    "+-": plus_minus_check,
    "><": qm_check,
    "][": bracket_check,
}

ALL_SYMBOLS = [
    "a",
    "b",
    "A",
    "B",
    " ",
    "+",
    "-",
    "?",
    "!",
    "[",
    "]",
    "<",
    ">",
    "1",
    "2",
]

EXTRA_SYMBOLS = [".", ",", "c", "C", "d", "D", "e", "E", "f", "F"]
