"""Tests for dual number eninge."""
import math
from dualnumber.engine import Dual


def test_unary_minus():
    """Tests unary minus operator."""
    d = Dual(2, -4)

    out = -d
    assert out.real == -2
    assert out.dual == 4


def test_add():
    """Tests addition of dual numbers."""
    d1 = Dual(2, -3)
    d2 = Dual(-5, 7)

    out = d1 + 11
    assert out.real == 13
    assert out.dual == -3

    out = 11 + d1
    assert out.real == 13
    assert out.dual == -3

    out = d1 + d2
    assert out.real == -3
    assert out.dual == 4


def test_sub():
    """Tests subtraction of dual numbers."""
    d1 = Dual(2, -3)
    d2 = Dual(-5, 7)

    out = d1 - 11
    assert out.real == -9
    assert out.dual == -3

    out = 11 - d1
    assert out.real == 9
    assert out.dual == 3

    out = d1 - d2
    assert out.real == 7
    assert out.dual == -10


def test_mul():
    """Tests multiplication of dual numbers."""
    d1 = Dual(2, -3)
    d2 = Dual(-5, 7)

    out = d1 * 11
    assert out.real == 22
    assert out.dual == -33

    out = 11 * d1
    assert out.real == 22
    assert out.dual == -33

    out = d1 * d2
    assert out.real == -10
    assert out.dual == 29


def test_div():
    """Tests division of dual numbers."""
    d1 = Dual(2, -3)
    d2 = Dual(-5, 7)

    out = d1 / 11
    assert out.real == 2 / 11
    assert out.dual == -3 / 11

    out = 11 / d1
    assert out.real == 11 / 2
    assert out.dual == -(11*(-3))/(2**2)

    out = d1 / d2
    assert out.real == 2 / -5
    assert out.dual == ((-3)*(-5)-2*7) / (-5)**2


def test_pow():
    """Tests exponentiation for dual numbers."""
    d1 = Dual(2, -3)
    d2 = Dual(-5, 7)

    out = d1**2
    assert out.real == 2**2
    assert out.dual == -12

    out = 2**d1
    assert out.real == 2**2
    assert out.dual == (2**2)*math.log(2)*(-3)

    out = d1**d2
    assert out.real == 2**(-5)
    assert out.dual == 2**(-5) * (((-3)/2)*(-5) + math.log(2)*7)


def test_sin():
    """Tests sine for dual numbers."""
    d = Dual(2, -3)

    out = d.sin()
    assert out.real == math.sin(2)
    assert out.dual == math.cos(2)*(-3)


def test_cos():
    """Tests cosine for dual numbers."""
    d = Dual(2, -3)

    out = d.cos()
    assert out.real == math.cos(2)
    assert out.dual == -math.sin(2)*(-3)


def test_tanh():
    """Tests tangens hyperbolicus for dual numbers."""
    d = Dual(2, -3)

    out = d.tanh()
    assert out.real == math.tanh(2)
    assert out.dual == (1.0 - math.tanh(2)**2)*(-3)


def test_exp():
    """Tests exponential function for dual numbers."""
    d = Dual(2, -3)

    out = d.exp()
    assert out.real == math.exp(2)
    assert out.dual == math.exp(2)*(-3)


def test_log():
    """Tests natural logarithm for dual numbers."""
    d = Dual(2, -3)

    out = d.log()
    assert out.real == math.log(2)
    assert out.dual == (1/2)*(-3)


def test_relu():
    """Tests relu activation function for dual numbers."""
    d1 = Dual(2, -3)

    out = d1.relu()
    assert out.real == 2
    assert out.dual == -3

    d2 = Dual(-5, 7)

    out = d2.relu()
    assert out.real == 0
    assert out.dual == 0


def test_less_than():
    """Tests less than operator (<) for dual numbers."""
    d1 = Dual(2, -3)
    d2 = Dual(-5, 7)

    out = d1 < d2
    assert out == False


def test_less_than_or_equal():
    """Tests less than or equal operator for dual numbers."""
    d1 = Dual(2, -3)
    d2 = Dual(-5, 7)

    out = d1 <= d2
    assert out == False

def test_equals():
    """Tests equals operator for dual numbers."""
    d1 = Dual(2, -3)
    d2 = Dual(-5, 7)

    out = d1 == d2
    assert out == False

def test_not_equals():
    """Tests not equals operator for dual numbers."""
    d1 = Dual(2, -3)
    d2 = Dual(-5, 7)

    out = d1 != d2
    assert out == True 


def test_greater_than():
    """Tests greater than operator for dual numbers."""
    d1 = Dual(2, -3)
    d2 = Dual(-5, 7)

    out = d1 > d2
    assert out == True 


def test_less_greater_or_equal():
    """Tests greater than or equal operator for dual numbers."""
    d1 = Dual(2, -3)
    d2 = Dual(-5, 7)

    out = d1 >= d2
    assert out == True 


def test_abs():
    """Tests abs() operator for dual numbers."""
    d = Dual(2, -3)

    out = abs(d)
    assert out.real == 2
    assert out.dual == 0

    out = d.abs()
    assert out.real == 2
    assert out.dual == 0



def test_conjugate():
    """Tests conjugation for dual numbers."""
    d = Dual(2, -3)

    out = d.conjugate()
    assert out.real == 2
    assert out.dual == 3