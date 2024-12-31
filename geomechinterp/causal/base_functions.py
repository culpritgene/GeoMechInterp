from typing import Callable

# gp = global position parity
all_binary_features = ["position_parity", "ab", "case", "12", "+-", "><", "?!", "]["]


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


all_binary_generators = {
    "position_parity": position_parity_check,  # no control!
    "ab": ab_f,
    "case": case_f,
    "12": f12_f,
    "+-": plus_minus_f,
    "><": qm_f,
    "][": bracket_f,
}

all_binary_checks = {
    "position_parity": position_parity_check,
    "ab": ab_check,
    "case": case_check,
    "12": f12_check,
    "+-": plus_minus_check,
    "><": qm_check,
    "][": bracket_check,
}
