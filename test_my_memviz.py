import pytest
import torch
import json
import os
from my_memviz import format_bytes, calculate_tensor_memory, GraphNode, Graph, extract_saved_tensors, traverse_graph, format_json, format_dot, format_image, dump_graph


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


def test_graph_node_with_call_stack():
    node = GraphNode(
        node_id=0,
        op_type="AddBackward",
        output_shape=[10, 20],
        call_stack=[
            {"file": "/path/to/file.py", "line": 42, "function": "forward", "code": "y = linear(x)"}
        ]
    )
    assert node.call_stack[0]["file"] == "/path/to/file.py"
    assert node.call_stack[0]["line"] == 42


def test_graph_empty():
    graph = Graph()
    assert graph.nodes == []
    assert graph.edges == []
    assert graph.total_saved_memory_bytes == 0


def test_graph_add_node():
    graph = Graph()
    node = GraphNode(node_id=0, op_type="AddBackward", output_shape=[10])
    graph.add_node(node)
    assert len(graph.nodes) == 1
    assert graph.nodes[0].op_type == "AddBackward"


def test_graph_add_edge():
    graph = Graph()
    graph.add_edge(0, 1)
    assert len(graph.edges) == 1
    assert graph.edges[0] == {"from": 0, "to": 1}


def test_graph_total_memory():
    graph = Graph()
    node1 = GraphNode(
        node_id=0,
        op_type="Op1",
        output_shape=[10],
        saved_memory_bytes=100
    )
    node2 = GraphNode(
        node_id=1,
        op_type="Op2",
        output_shape=[20],
        saved_memory_bytes=200
    )
    graph.add_node(node1)
    graph.add_node(node2)
    assert graph.total_saved_memory_bytes == 300


def test_extract_saved_tensors_linear():
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 30)
    y = linear(x)
    
    saved = extract_saved_tensors(y.grad_fn)
    assert len(saved) >= 1
    assert any(st["name"] in ["mat1", "weight", "bias"] for st in saved)


def test_extract_saved_tensors_relu():
    x = torch.randn(5, 10, requires_grad=True)
    y = torch.relu(x)
    
    saved = extract_saved_tensors(y.grad_fn)
    assert len(saved) == 1
    assert saved[0]["name"] == "result"


def test_traverse_graph_simple():
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 30)
    y = linear(x)
    
    graph = Graph()
    traverse_graph(y, x, graph)
    
    assert len(graph.nodes) >= 1
    assert any("AddmmBackward" in node.op_type for node in graph.nodes)


def test_traverse_graph_multi_layer():
    x = torch.randn(10, 20, requires_grad=True)
    model = torch.nn.Sequential(
        torch.nn.Linear(20, 30),
        torch.nn.ReLU(),
        torch.nn.Linear(30, 10)
    )
    y = model(x)
    
    graph = Graph()
    traverse_graph(y, x, graph)
    
    assert len(graph.nodes) >= 3
    assert len(graph.edges) >= 2


def test_traverse_graph_no_path():
    x = torch.randn(10, 20, requires_grad=True)
    y = torch.randn(10, 20, requires_grad=True)
    
    graph = Graph()
    traverse_graph(y, x, graph)
    
    assert len(graph.nodes) == 0


def test_format_json_basic():
    graph = Graph()
    node = GraphNode(
        node_id=0,
        op_type="AddBackward",
        output_shape=[10, 20],
        saved_tensors=[{"name": "result", "shape": [10, 20], "dtype": "float32", "size_bytes": 800, "size_formatted": "800"}],
        saved_memory_bytes=800
    )
    graph.add_node(node)
    graph.add_edge(0, 0)
    
    result = format_json(graph)
    data = json.loads(result)
    
    assert data["summary"]["num_nodes"] == 1
    assert data["summary"]["total_saved_memory_bytes"] == 800
    assert len(data["nodes"]) == 1
    assert data["nodes"][0]["op_type"] == "AddBackward"
    assert data["edges"][0] == {"from": 0, "to": 0}


def test_format_json_with_call_stack():
    graph = Graph()
    node = GraphNode(
        node_id=0,
        op_type="AddBackward",
        output_shape=[10, 20],
        call_stack=[
            {"file": "/path/model.py", "line": 15, "function": "forward", "code": "y = self.linear(x)"}
        ]
    )
    graph.add_node(node)
    
    result = format_json(graph, show_memory=False)
    data = json.loads(result)
    
    assert "call_stack" in data["nodes"][0]
    assert data["nodes"][0]["call_stack"][0]["file"] == "/path/model.py"
    assert data["nodes"][0]["call_stack"][0]["line"] == 15


def test_format_json_no_memory():
    graph = Graph()
    node = GraphNode(
        node_id=0,
        op_type="AddBackward",
        output_shape=[10, 20]
    )
    graph.add_node(node)
    
    result = format_json(graph, show_memory=False)
    data = json.loads(result)
    
    assert data["summary"]["num_nodes"] == 1
    assert "total_saved_memory_bytes" not in data["summary"]
    assert "saved_tensors" not in data["nodes"][0]


def test_format_dot_basic():
    graph = Graph()
    node = GraphNode(
        node_id=0,
        op_type="AddBackward",
        output_shape=[10, 20]
    )
    graph.add_node(node)
    graph.add_edge(0, 0)
    
    result = format_dot(graph, show_memory=False)
    
    assert "digraph computation_graph" in result
    assert "rankdir=LR" in result
    assert "AddBackward" in result
    assert "node0" in result


def test_format_dot_with_memory():
    graph = Graph()
    node = GraphNode(
        node_id=0,
        op_type="ReluBackward",
        output_shape=[10, 20],
        saved_memory_bytes=800
    )
    graph.add_node(node)
    
    result = format_dot(graph, show_memory=True)
    
    assert "memory: 800" in result


def test_format_dot_with_call_stack():
    graph = Graph()
    node = GraphNode(
        node_id=0,
        op_type="AddBackward",
        output_shape=[10, 20],
        call_stack=[
            {"file": "model.py", "line": 15, "function": "forward", "code": "y = self.linear(x)"}
        ]
    )
    graph.add_node(node)
    
    result = format_dot(graph, show_memory=False)
    
    assert "model.py:15" in result
    assert "forward" in result


def test_format_image_png():
    graph = Graph()
    node = GraphNode(
        node_id=0,
        op_type="AddBackward",
        output_shape=[10, 20]
    )
    graph.add_node(node)
    
    result = format_image(graph, "test_output.png", format="png")
    
    try:
        import graphviz
        assert result == "test_output.png"
        assert os.path.exists("test_output.png")
        os.remove("test_output.png")
    except ImportError:
        assert result is None


def test_format_image_svg():
    graph = Graph()
    node = GraphNode(
        node_id=0,
        op_type="AddBackward",
        output_shape=[10, 20]
    )
    graph.add_node(node)
    
    result = format_image(graph, "test_output.svg", format="svg")
    
    try:
        import graphviz
        assert result == "test_output.svg"
        assert os.path.exists("test_output.svg")
        os.remove("test_output.svg")
    except ImportError:
        assert result is None


def test_dump_graph_json():
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 30)
    y = linear(x)
    
    graph = dump_graph(y, x, format="json")
    result = graph.render("json")
    data = json.loads(result)
    
    assert data["summary"]["num_nodes"] >= 1
    assert "total_saved_memory_bytes" in data["summary"]


def test_dump_graph_dot():
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 30)
    y = linear(x)
    
    graph = dump_graph(y, x, format="dot")
    result = graph.render("dot")
    
    assert "digraph computation_graph" in result
    assert "rankdir=LR" in result


def test_dump_graph_to_file():
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 30)
    y = linear(x)
    
    dump_graph(y, x, format="json", output_file="test_graph")
    
    assert os.path.exists("test_graph.json")
    with open("test_graph.json", "r") as f:
        data = json.loads(f.read())
    assert data["summary"]["num_nodes"] >= 1
    os.remove("test_graph.json")


def test_dump_graph_multiple_formats():
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 30)
    y = linear(x)
    
    graph = dump_graph(y, x, format=["json", "dot"], output_file="test_multi")
    
    assert os.path.exists("test_multi.json")
    assert os.path.exists("test_multi.dot")
    os.remove("test_multi.json")
    os.remove("test_multi.dot")


def test_dump_graph_disconnected_tensors():
    x = torch.randn(10, 20, requires_grad=True)
    y = torch.randn(10, 20, requires_grad=True)
    
    with pytest.warns(UserWarning):
        result = dump_graph(y, x, format="json")
    
    assert result is not None


def test_dump_graph_non_tensor_input():
    y = torch.randn(10, 20, requires_grad=True)
    
    with pytest.warns(UserWarning):
        result = dump_graph(y, "not a tensor", format="json")
    
    assert result is None


def test_dump_graph_no_requires_grad():
    x = torch.randn(10, 20, requires_grad=False)
    y = torch.randn(10, 20, requires_grad=True)
    
    with pytest.warns(UserWarning):
        result = dump_graph(y, x, format="json")


def test_integration_multi_layer_network():
    x = torch.randn(32, 784, requires_grad=True)
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    )
    y = model(x)
    
    graph = dump_graph(y, x, format="json", show_memory=True)
    data = json.loads(graph.render("json"))
    
    assert data["summary"]["num_nodes"] >= 5
    assert data["summary"]["total_saved_memory_bytes"] > 0


def test_integration_residual_connection():
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 20)
    
    y = linear(x)
    y = y + x
    
    graph = dump_graph(y, x, format="json", show_memory=True)
    data = json.loads(graph.render("json"))
    
    assert data["summary"]["num_nodes"] >= 1
    assert any("AddBackward" in node["op_type"] for node in data["nodes"])


def test_integration_memory_tracking():
    x = torch.randn(100, 100, requires_grad=True)
    linear = torch.nn.Linear(100, 100)
    y = linear(x)
    
    graph = dump_graph(y, x, format="json", show_memory=True)
    data = json.loads(graph.render("json"))
    
    assert data["summary"]["total_saved_memory_bytes"] > 0
    
    for node in data["nodes"]:
        if len(node.get("saved_tensors", [])) > 0:
            for st in node["saved_tensors"]:
                assert "size_bytes" in st
                assert "size_formatted" in st
                assert st["size_bytes"] > 0