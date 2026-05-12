import pytest
from my_memviz import format_bytes


def test_format_bytes_zero():
    assert format_bytes(0) == "0"


def test_format_bytes_small():
    assert format_bytes(123) == "123"


def test_format_bytes_thousands():
    assert format_bytes(1234567) == "1,234,567"


def test_format_bytes_large():
    assert format_bytes(9876543210) == "9,876,543,210"