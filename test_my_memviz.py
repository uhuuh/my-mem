import pytest
import torch
import json
import os
from my_memviz import format_bytes, calculate_tensor_memory, GraphNode, Graph, extract_saved_tensors, format_json, format_dot, format_image, dump_graph, _extract_call_stack, _capture_tensor_info, _find_end_nodes, _build_subgraph


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


def test_dump_graph_default_file_naming():
    pid = os.getpid()
    
    with dump_graph(format="json"):
        x = torch.randn(10, 20, requires_grad=True)
        linear = torch.nn.Linear(20, 30)
        y = linear(x)
    
    expected_file = f"dump_{pid}.json"
    assert os.path.exists(expected_file)
    
    with open(expected_file, "r") as f:
        data = json.loads(f.read())
    assert data["summary"]["num_nodes"] >= 1
    
    os.remove(expected_file)


def test_dump_graph_json():
    with dump_graph(format="json", output_file="test_json_dump"):
        x = torch.randn(10, 20, requires_grad=True)
        linear = torch.nn.Linear(20, 30)
        y = linear(x)
    
    assert os.path.exists("test_json_dump.json")
    with open("test_json_dump.json", "r") as f:
        data = json.loads(f.read())
    assert data["summary"]["num_nodes"] >= 1
    assert "total_saved_memory_bytes" in data["summary"]
    os.remove("test_json_dump.json")


def test_dump_graph_dot():
    with dump_graph(format="dot", output_file="test_dot_dump"):
        x = torch.randn(10, 20, requires_grad=True)
        linear = torch.nn.Linear(20, 30)
        y = linear(x)
    
    assert os.path.exists("test_dot_dump.dot")
    with open("test_dot_dump.dot", "r") as f:
        content = f.read()
    assert "digraph computation_graph" in content
    assert "rankdir=LR" in content
    os.remove("test_dot_dump.dot")


def test_dump_graph_to_file():
    with dump_graph(format="json", output_file="test_graph"):
        x = torch.randn(10, 20, requires_grad=True)
        linear = torch.nn.Linear(20, 30)
        y = linear(x)
    
    assert os.path.exists("test_graph.json")
    with open("test_graph.json", "r") as f:
        data = json.loads(f.read())
    assert data["summary"]["num_nodes"] >= 1
    os.remove("test_graph.json")


def test_dump_graph_multiple_formats():
    with dump_graph(format=["json", "dot"], output_file="test_multi"):
        x = torch.randn(10, 20, requires_grad=True)
        linear = torch.nn.Linear(20, 30)
        y = linear(x)
    
    assert os.path.exists("test_multi.json")
    assert os.path.exists("test_multi.dot")
    os.remove("test_multi.json")
    os.remove("test_multi.dot")


def test_dump_graph_empty_block():
    with pytest.warns(UserWarning):
        with dump_graph(format="json", output_file="test_empty"):
            pass
    
    assert os.path.exists("test_empty.json")
    with open("test_empty.json", "r") as f:
        data = json.loads(f.read())
    assert data["summary"]["num_nodes"] == 0
    os.remove("test_empty.json")


def test_dump_graph_non_requires_grad_leaf():
    with dump_graph(format="json", output_file="test_no_grad"):
        x = torch.randn(10, 20, requires_grad=False)
        y = x + 1
    
    assert os.path.exists("test_no_grad.json")
    with open("test_no_grad.json", "r") as f:
        data = json.loads(f.read())
    os.remove("test_no_grad.json")


def test_extract_call_stack():
    import traceback
    
    stack = traceback.extract_stack()
    result = _extract_call_stack(stack)
    
    assert len(result) > 0
    assert "file" in result[0]
    assert "line" in result[0]
    assert "function" in result[0]
    assert "code" in result[0]


def test_integration_multi_layer_network():
    with dump_graph(format="json", output_file="test_integration_multi", show_memory=True):
        x = torch.randn(32, 784, requires_grad=True)
        model = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )
        y = model(x)
    
    assert os.path.exists("test_integration_multi.json")
    with open("test_integration_multi.json", "r") as f:
        data = json.loads(f.read())
    
    assert data["summary"]["num_nodes"] >= 5
    assert data["summary"]["total_saved_memory_bytes"] > 0
    os.remove("test_integration_multi.json")


def test_integration_residual_connection():
    with dump_graph(format="json", output_file="test_residual", show_memory=True):
        x = torch.randn(10, 20, requires_grad=True)
        linear = torch.nn.Linear(20, 20)
        
        y = linear(x)
        y = y + x
    
    assert os.path.exists("test_residual.json")
    with open("test_residual.json", "r") as f:
        data = json.loads(f.read())
    
    assert data["summary"]["num_nodes"] >= 1
    os.remove("test_residual.json")


def test_integration_memory_tracking():
    with dump_graph(format="json", output_file="test_memory", show_memory=True):
        x = torch.randn(100, 100, requires_grad=True)
        linear = torch.nn.Linear(100, 100)
        y = linear(x)
    
    assert os.path.exists("test_memory.json")
    with open("test_memory.json", "r") as f:
        data = json.loads(f.read())
    
    assert data["summary"]["total_saved_memory_bytes"] > 0
    
    for node in data["nodes"]:
        if len(node.get("saved_tensors", [])) > 0:
            for st in node["saved_tensors"]:
                assert "size_bytes" in st
                assert "size_formatted" in st
                assert st["size_bytes"] > 0
    os.remove("test_memory.json")


def test_capture_tensor_info():
    x = torch.randn(10, 20, requires_grad=True)
    info = _capture_tensor_info(x)
    
    assert info['id'] == id(x)
    assert info['requires_grad'] == True
    assert info['shape'] == [10, 20]
    assert 'dtype' in info
    assert 'call_stack' in info
    
    y = torch.randn(5, 10, requires_grad=False)
    info_y = _capture_tensor_info(y)
    
    assert info_y['id'] == id(y)
    assert info_y['requires_grad'] == False
    assert info_y['shape'] == [5, 10]


def test_find_end_nodes():
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 30)
    y = linear(x)
    z = y * 2
    
    captured = [_capture_tensor_info(y), _capture_tensor_info(z)]
    
    end_nodes = _find_end_nodes(captured)
    
    assert len(end_nodes) >= 1
    assert any(t['tensor'] is z for t in end_nodes)


def test_build_subgraph():
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 30)
    y = linear(x)
    
    captured = [_capture_tensor_info(y)]
    end_nodes = _find_end_nodes(captured)
    
    graph = _build_subgraph(end_nodes, captured)
    
    assert isinstance(graph, Graph)
    assert len(graph.nodes) >= 1


def test_dump_graph_context_manager_basic():
    with dump_graph(format="json", output_file="test_ctx"):
        x = torch.randn(10, 20, requires_grad=True)
        linear = torch.nn.Linear(20, 30)
        y = linear(x)
    
    assert os.path.exists("test_ctx.json")
    with open("test_ctx.json", "r") as f:
        data = json.loads(f.read())
    assert data["summary"]["num_nodes"] >= 1
    assert "call_stack" in data["nodes"][0]
    os.remove("test_ctx.json")


def test_integration_context_manager_multi_layer():
    with dump_graph(format="json", output_file="test_integration"):
        x = torch.randn(32, 784, requires_grad=True)
        model = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )
        y = model(x)
    
    assert os.path.exists("test_integration.json")
    with open("test_integration.json", "r") as f:
        data = json.loads(f.read())
    
    assert data["summary"]["num_nodes"] >= 5
    assert data["summary"]["total_saved_memory_bytes"] > 0
    
    for node in data["nodes"]:
        assert "call_stack" in node
        if len(node["call_stack"]) > 0:
            assert "file" in node["call_stack"][0]
            assert "line" in node["call_stack"][0]
    
    os.remove("test_integration.json")


def test_dump_graph_tensor_operations():
    with dump_graph(format="json", output_file="test_tensor_ops"):
        x = torch.randn(10, 20, requires_grad=True)
        y = x + 1
        z = y * 2
        w = z - 0.5
    
    assert os.path.exists("test_tensor_ops.json")
    with open("test_tensor_ops.json", "r") as f:
        data = json.loads(f.read())
    
    assert data["summary"]["num_nodes"] >= 3
    op_types = [n["op_type"] for n in data["nodes"]]
    assert "AddBackward0" in op_types
    assert "MulBackward0" in op_types
    assert "SubBackward0" in op_types
    
    os.remove("test_tensor_ops.json")


def test_dump_graph_torch_functions():
    with dump_graph(format="json", output_file="test_torch_funcs"):
        x = torch.randn(10, 20, requires_grad=True)
        y = torch.relu(x)
        z = torch.sigmoid(y)
        w = torch.tanh(z)
    
    assert os.path.exists("test_torch_funcs.json")
    with open("test_torch_funcs.json", "r") as f:
        data = json.loads(f.read())
    
    assert data["summary"]["num_nodes"] >= 3
    op_types = [n["op_type"] for n in data["nodes"]]
    assert "ReluBackward0" in op_types
    assert "SigmoidBackward0" in op_types
    assert "TanhBackward0" in op_types
    
    os.remove("test_torch_funcs.json")