"""A class for dual numbers.

Basic implementation of dual numbers in Python.

    Typical usage example:

    d1 = Dual(1, 2)
    d2 = Dual(3, 4)
    d3 = d1 + d2
    d4 = d3 - d2
    d5 = d4 * d3
    d6 = d5 / d4
    d7 = d6.sin()
    d8 = d7.cos()
    d9 = d9.tanh()
    d10 = d9.ln()
    d11 = d10.exp()
    d12 = d11**d10
    d12 = d11.relu()
"""
from __future__ import annotations
from typing import Union

from math import cos, sin, tanh, exp, log


class Dual:
    r"""Dual is a class for dual number arithmetic.

    This class for dual numbers implements basic arithmetic operations
    as well as more advanced operations for dual numbers.

    Attributes:
        real: Real part of dual number.
        dual: Dual part of dual number.
    """

    def __init__(self, real: Union[float, int], dual: Union[float, int] = 0.0) -> None:
        self.real = real
        self.dual = dual

    def sin(self) -> Dual:
        r"""Implements sine for dual number."""
        real = sin(self.real)
        dual = cos(self.real) * self.dual
        return Dual(real=real, dual=dual)

    def cos(self) -> Dual:
        r"""Implements cosine for dual number."""
        real = cos(self.real)
        dual = -sin(self.real) * self.dual
        return Dual(real=real, dual=dual)

    def tanh(self) -> Dual:
        r"""Implements tangens hyperbolicus for dual number."""
        real = tanh(self.real)
        dual = (1.0 - real**2) * self.dual
        return Dual(real=real, dual=dual)

    def exp(self) -> Dual:
        r"""Implements exponentiation for dual number."""
        real = exp(self.real)
        dual = real * self.dual
        return Dual(real=real, dual=dual)

    def log(self) -> Dual:
        r"""Implements logarithm for dual number."""
        assert self.real != 0, f"Real part of denominator must be nonnegative."
        real = log(self.real)
        dual = self.dual / self.real
        return Dual(real=real, dual=dual)

    def relu(self) -> Dual:
        r"""Implements ReLU activation function for dual number."""
        if self.real > 0:
            real = self.real
            dual = self.dual
        else:
            real = 0.0
            dual = 0.0
        return Dual(real=real, dual=dual)

    def __pow__(self, power: Union[Dual, float, int]) -> Dual:
        r"""Implements power operator for dual number."""
        assert self.real != 0, f"Real part of denominator must be nonnegative."
        if isinstance(power, Dual):
            other = power
            real = self.real**other.real
            dual = real * (
                (self.dual / self.real) * other.real + log(self.real) * other.dual
            )
            return Dual(real=real, dual=dual)
        else:
            real = self.real**power
            dual = real * (self.dual / self.real) * power
            return Dual(real=real, dual=dual)

    def __rpow__(self, other: Union[float, int]) -> Dual:
        r"""Implements reverse power operator for dual number."""
        # other is the base here without dual part
        assert other > 0, f"Base must be nonnegative but is {other}"
        other = Dual(real=other)
        return other**self

    def __add__(self, other: Dual) -> Dual:
        r"""Adds two dual numbers.

        The addition of dual numbers is implemented as follows:

        .. math::
            (a + \epsilon b) + (c + \epsilon d) = (a + c) + \epsilon (b + d)

        Args:
            self: A dual number instance.
            other: A dual number instance.

        Returns:
            A dual number.
        """
        other = other if isinstance(other, Dual) else Dual(real=other)
        return Dual(self.real + other.real, self.dual + other.dual)

    def __radd__(self, other: Dual) -> Dual:
        r"""Implements reverse addition."""
        return self + other

    def __sub__(self, other: Dual) -> Dual:
        r"""Subtracts two dual numbers.

        The subtraction of dual numbers is implemented as follows:

        .. math::
            (a + \epsilon b) - (c + \epsilon d) = (a - c) + \epsilon (b - d)

        Args:
            self: A dual number instance.
            other: A dual number instance.

        Returns:
            A dual number.
        """
        other = other if isinstance(other, Dual) else Dual(real=other)
        return Dual(self.real - other.real, self.dual - other.dual)

    def __rsub__(self, other: Dual) -> Dual:
        r"""Implements reverse subtraction."""
        return other + (-self)

    def __mul__(self, other: Dual) -> Dual:
        r"""Multiplies two dual numbers.

        The multiplication of dual numbers is implemented as follows:

        .. math::
            (a + \epsilon b) \cdot (c + \epsilon d) = a \cdot c + \epsilon (a \cdot d + b \cdot c)

        Args:
            self: A dual number instance.
            other: A dual number instance.

        Returns:
            A dual number.
        """
        other = other if isinstance(other, Dual) else Dual(real=other)
        real = self.real * other.real
        dual = self.real * other.dual + self.dual * other.real
        return Dual(real=real, dual=dual)

    def __rmul__(self, other: Dual) -> Dual:
        return self * other

    def __truediv__(self, other: Dual) -> Dual:
        r"""Divides two dual numbers.

        The division of dual numbers is given by:

        .. math::
            \frac{a + \epsilon b}{c + \epsilon d} = \frac{a}{c} + \epsilon \frac{b \cdot c - a \cdot d}{c^2}
        Args:
            self: A dual number instance.
            other: A dual number instance.

        Returns:
            A dual number.

        Raises:
            Error if dual number in denominator is zero.
        """
        assert other.real != 0, f"Real part of denominator must be nonnegative."
        other = other if isinstance(other, Dual) else Dual(real=other)
        real = self.real / other.real
        dual = (self.dual * other.real - self.real * other.dual) / (
            other.real * other.real
        )
        return Dual(real=real, dual=dual)

    def __rtruediv__(self, other: Union[float, int]) -> Dual:
        r"""Reverse division."""
        return Dual(real=other) / self

    def __neg__(self) -> Dual:
        r"""Implements the unary minus operator."""
        return Dual(-self.real, -self.dual)

    def __lt__(self, other: Dual) -> bool:
        r"""Implements 'less than' (<) operator."""
        return self.real < other.real

    def __le__(self, other: Dual) -> bool:
        r"""Implements 'less than or equal' (<=) operator."""
        return self.real <= other.real

    def __eq__(self, other: Dual) -> bool:
        r"""Implements 'equals' (=) operator.

        Dual numbers are equal if their real parts are equal.
        """
        return self.real == other.real

    def __ne__(self, other: Dual) -> bool:
        r"""Implements 'not equals' (!=) operator.

        Dual numbers are inequal if their real parts are inequal.
        """
        return not self.__eq__(other)

    def __ge__(self, other: Dual) -> bool:
        r"""Implements 'less than' (>) operator."""
        return self.real > other.real

    def __gt__(self, other: Dual) -> bool:
        r"""Implements 'less than or equal' (>=) operator."""
        return self.real >= other.real

    def conjugate(self) -> Dual:
        r"""Conjugates dual number."""
        return Dual(real=self.real, dual=-self.dual)

    def __abs__(self) -> Dual:
        r"""Computes absolute value."""
        return Dual(real=self.real, dual=0.0)

    def abs(self) -> Dual:
        r"""Computes absolute value."""
        return self.__abs__()

    def __str__(self) -> str:
        return f"({self.real}, {self.dual})"

    def __repr__(self) -> str:
        return f"({self.real}, {self.dual})"
