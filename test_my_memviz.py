import pytest
import torch
from my_memviz import format_bytes, calculate_tensor_memory, GraphNode


def test_format_bytes_zero():
    assert format_bytes(0) == "0"


def test_format_bytes_small():
    assert format_bytes(123) == "123"


def test_format_bytes_thousands():
    assert format_bytes(1234567) == "1,234,567"


def test_format_bytes_large():
    assert format_bytes(9876543210) == "9,876,543,210"


def test_calculate_tensor_memory_float32():
    tensor = torch.randn(10, 20, dtype=torch.float32)
    assert calculate_tensor_memory(tensor) == 10 * 20 * 4


def test_calculate_tensor_memory_float16():
    tensor = torch.randn(5, 10, dtype=torch.float16)
    assert calculate_tensor_memory(tensor) == 5 * 10 * 2


def test_calculate_tensor_memory_int64():
    tensor = torch.randint(0, 10, (8, 8), dtype=torch.int64)
    assert calculate_tensor_memory(tensor) == 8 * 8 * 8


def test_graph_node_basic():
    node = GraphNode(
        node_id=0,
        op_type="AddmmBackward",
        output_shape=[10, 20]
    )
    assert node.node_id == 0
    assert node.op_type == "AddmmBackward"
    assert node.output_shape == [10, 20]
    assert node.saved_tensors == []
    assert node.saved_memory_bytes == 0


def test_graph_node_with_saved_tensors():
    node = GraphNode(
        node_id=1,
        op_type="ReluBackward",
        output_shape=[5, 10],
        saved_tensors=[
            {"name": "result", "shape": [5, 10], "dtype": "float32", "size_bytes": 200, "size_formatted": "200"}
        ],
        saved_memory_bytes=200
    )
    assert len(node.saved_tensors) == 1
    assert node.saved_tensors[0]["name"] == "result"