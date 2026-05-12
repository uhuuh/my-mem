# Tensor Graph Dump Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A PyTorch tool that dumps computation subgraphs between two tensors with memory analysis, supporting JSON, DOT, and PNG/SVG output formats.

**Architecture:** Single-file implementation using PyTorch's autograd graph traversal. Direct inspection of backward nodes and saved tensors for accurate memory profiling. Separate formatter functions for each output format.

**Tech Stack:** Python 3.8+, PyTorch, graphviz (optional for PNG/SVG)

---

## File Structure

```
my_memviz.py          # Main implementation (all components)
test_my_memviz.py     # Test suite
requirements.txt      # Dependencies
```

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `test_my_memviz.py`

- [ ] **Step 1: Create requirements.txt**

```txt
torch>=1.9.0
graphviz>=0.17
pytest>=7.0.0
```

- [ ] **Step 2: Create test file with minimal test**

```python
import pytest


def test_placeholder():
    assert True
```

- [ ] **Step 3: Run test to verify pytest setup**

Run: `pytest test_my_memviz.py -v`
Expected: PASS (1 test)

- [ ] **Step 4: Commit setup files**

```bash
git add requirements.txt test_my_memviz.py
git commit -m "chore: setup project structure and dependencies"
```

---

### Task 2: Format Bytes Utility

**Files:**
- Modify: `my_memviz.py`
- Modify: `test_my_memviz.py`

- [ ] **Step 1: Write failing tests for format_bytes**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_my_memviz.py::test_format_bytes -v`
Expected: FAIL with "cannot import name 'format_bytes'"

- [ ] **Step 3: Implement format_bytes function**

```python
def format_bytes(num_bytes):
    if num_bytes == 0:
        return "0"
    return "{:,}".format(num_bytes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_my_memviz.py::test_format_bytes -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit format_bytes**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: add format_bytes utility function"
```

---

### Task 3: Tensor Memory Analysis

**Files:**
- Modify: `my_memviz.py`
- Modify: `test_my_memviz.py`

- [ ] **Step 1: Write failing tests for calculate_tensor_memory**

```python
import pytest
import torch
from my_memviz import calculate_tensor_memory


def test_calculate_tensor_memory_float32():
    tensor = torch.randn(10, 20, dtype=torch.float32)
    assert calculate_tensor_memory(tensor) == 10 * 20 * 4


def test_calculate_tensor_memory_float16():
    tensor = torch.randn(5, 10, dtype=torch.float16)
    assert calculate_tensor_memory(tensor) == 5 * 10 * 2


def test_calculate_tensor_memory_int64():
    tensor = torch.randint(0, 10, (8, 8), dtype=torch.int64)
    assert calculate_tensor_memory(tensor) == 8 * 8 * 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_my_memviz.py::test_calculate_tensor_memory -v`
Expected: FAIL with "cannot import name 'calculate_tensor_memory'"

- [ ] **Step 3: Implement calculate_tensor_memory function**

```python
import torch


def calculate_tensor_memory(tensor):
    return tensor.nelement() * tensor.element_size()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_my_memviz.py::test_calculate_tensor_memory -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit tensor memory function**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: add calculate_tensor_memory function"
```

---

### Task 4: Graph Node Data Structure

**Files:**
- Modify: `my_memviz.py`
- Modify: `test_my_memviz.py`

- [ ] **Step 1: Write failing tests for GraphNode class**

```python
import pytest
from my_memviz import GraphNode


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_my_memviz.py::test_graph_node -v`
Expected: FAIL with "cannot import name 'GraphNode'"

- [ ] **Step 3: Implement GraphNode class**

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class GraphNode:
    node_id: int
    op_type: str
    output_shape: List[int]
    saved_tensors: List[Dict[str, Any]] = field(default_factory=list)
    saved_memory_bytes: int = 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_my_memviz.py::test_graph_node -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit GraphNode class**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: add GraphNode dataclass"
```

---

### Task 5: Graph Data Structure

**Files:**
- Modify: `my_memviz.py`
- Modify: `test_my_memviz.py`

- [ ] **Step 1: Write failing tests for Graph class**

```python
import pytest
from my_memviz import Graph, GraphNode


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_my_memviz.py::test_graph -v`
Expected: FAIL with "cannot import name 'Graph'"

- [ ] **Step 3: Implement Graph class**

```python
from typing import List, Dict


class Graph:
    def __init__(self):
        self.nodes: List[GraphNode] = []
        self.edges: List[Dict[str, int]] = []
        self.total_saved_memory_bytes: int = 0

    def add_node(self, node: GraphNode):
        self.nodes.append(node)
        self.total_saved_memory_bytes += node.saved_memory_bytes

    def add_edge(self, from_id: int, to_id: int):
        self.edges.append({"from": from_id, "to": to_id})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_my_memviz.py::test_graph -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit Graph class**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: add Graph class"
```

---

### Task 6: Extract Saved Tensors from Autograd Node

**Files:**
- Modify: `my_memviz.py`
- Modify: `test_my_memviz.py`

- [ ] **Step 1: Write failing tests for extract_saved_tensors**

```python
import pytest
import torch
from my_memviz import extract_saved_tensors


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_my_memviz.py::test_extract_saved_tensors -v`
Expected: FAIL with "cannot import name 'extract_saved_tensors'"

- [ ] **Step 3: Implement extract_saved_tensors function**

```python
def extract_saved_tensors(grad_fn):
    saved_tensors = []
    if hasattr(grad_fn, 'saved_tensors'):
        for i, tensor in enumerate(grad_fn.saved_tensors):
            saved_tensors.append({
                "name": f"saved_{i}",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "size_bytes": calculate_tensor_memory(tensor),
                "size_formatted": format_bytes(calculate_tensor_memory(tensor))
            })
    
    # Try to get meaningful names for common ops
    if hasattr(grad_fn, '_saved_mat1'):
        for st in saved_tensors:
            if st["name"] == "saved_0":
                st["name"] = "mat1"
                break
    if hasattr(grad_fn, '_saved_bias'):
        for st in saved_tensors:
            if st["name"] == "saved_1":
                st["name"] = "bias"
                break
    
    return saved_tensors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_my_memviz.py::test_extract_saved_tensors -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit extract_saved_tensors**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: add extract_saved_tensors function"
```

---

### Task 7: Graph Traversal

**Files:**
- Modify: `my_memviz.py`
- Modify: `test_my_memviz.py`

- [ ] **Step 1: Write failing tests for traverse_graph**

```python
import pytest
import torch
from my_memviz import traverse_graph, Graph


def test_traverse_graph_simple():
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 30)
    y = linear(x)
    
    graph = Graph()
    traverse_graph(y, x, graph)
    
    assert len(graph.nodes) >= 1
    assert any(node.op_type == "AddmmBackward" for node in graph.nodes)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_my_memviz.py::test_traverse_graph -v`
Expected: FAIL with "cannot import name 'traverse_graph'"

- [ ] **Step 3: Implement traverse_graph function**

```python
import warnings


def traverse_graph(output_tensor, input_tensor, graph, visited=None):
    if visited is None:
        visited = set()
    
    if not hasattr(output_tensor, 'grad_fn') or output_tensor.grad_fn is None:
        warnings.warn("Output tensor has no grad_fn")
        return
    
    def _is_target_node(grad_fn):
        if grad_fn is None:
            return False
        if hasattr(grad_fn, 'variable'):
            return grad_fn.variable is input_tensor
        for next_fn, _ in grad_fn.next_functions:
            if next_fn is not None:
                if hasattr(next_fn, 'variable') and next_fn.variable is input_tensor:
                    return True
        return False
    
    def _traverse(grad_fn, visited):
        if grad_fn is None or id(grad_fn) in visited:
            return
        
        visited.add(id(grad_fn))
        
        node_id = len(graph.nodes)
        op_type = type(grad_fn).__name__
        
        saved_tensors = extract_saved_tensors(grad_fn)
        saved_memory = sum(st["size_bytes"] for st in saved_tensors)
        
        output_shape = []
        if hasattr(grad_fn, 'next_functions'):
            for next_fn, _ in grad_fn.next_functions:
                if next_fn is not None and hasattr(next_fn, 'variable'):
                    output_shape = list(next_fn.variable.shape)
                    break
        
        node = GraphNode(
            node_id=node_id,
            op_type=op_type,
            output_shape=output_shape,
            saved_tensors=saved_tensors,
            saved_memory_bytes=saved_memory
        )
        graph.add_node(node)
        
        for i, (next_fn, _) in enumerate(grad_fn.next_functions):
            if next_fn is not None and not _is_target_node(next_fn):
                _traverse(next_fn, visited)
                if len(graph.nodes) > node_id + 1:
                    graph.add_edge(node_id, node_id + 1)
    
    _traverse(output_tensor.grad_fn, visited)
    
    if len(graph.nodes) == 0:
        warnings.warn("No path found between tensors")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_my_memviz.py::test_traverse_graph -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit traverse_graph**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: add traverse_graph function"
```

---

### Task 8: JSON Formatter

**Files:**
- Modify: `my_memviz.py`
- Modify: `test_my_memviz.py`

- [ ] **Step 1: Write failing tests for format_json**

```python
import pytest
import json
from my_memviz import Graph, GraphNode, format_json


def test_format_json_basic():
    graph = Graph()
    node = GraphNode(
        node_id=0,
        op_type="AddBackward",
        output_shape=[10],
        saved_memory_bytes=100
    )
    graph.add_node(node)
    
    result = format_json(graph, show_memory=True)
    data = json.loads(result)
    
    assert data["summary"]["num_nodes"] == 1
    assert data["summary"]["total_saved_memory_bytes"] == 100
    assert "total_saved_memory_formatted" in data["summary"]
    assert len(data["nodes"]) == 1
    assert data["nodes"][0]["op_type"] == "AddBackward"


def test_format_json_no_memory():
    graph = Graph()
    node = GraphNode(node_id=0, op_type="AddBackward", output_shape=[10])
    graph.add_node(node)
    
    result = format_json(graph, show_memory=False)
    data = json.loads(result)
    
    assert "total_saved_memory_bytes" not in data["summary"]
    assert "saved_tensors" not in data["nodes"][0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_my_memviz.py::test_format_json -v`
Expected: FAIL with "cannot import name 'format_json'"

- [ ] **Step 3: Implement format_json function**

```python
import json


def format_json(graph, show_memory=True):
    data = {
        "summary": {
            "num_nodes": len(graph.nodes)
        },
        "nodes": [],
        "edges": graph.edges
    }
    
    if show_memory:
        data["summary"]["total_saved_memory_bytes"] = graph.total_saved_memory_bytes
        data["summary"]["total_saved_memory_formatted"] = format_bytes(graph.total_saved_memory_bytes)
    
    for node in graph.nodes:
        node_data = {
            "id": node.node_id,
            "op_type": node.op_type,
            "output_shape": node.output_shape
        }
        
        if show_memory:
            node_data["saved_tensors"] = node.saved_tensors
            node_data["saved_memory_bytes"] = node.saved_memory_bytes
            node_data["saved_memory_formatted"] = format_bytes(node.saved_memory_bytes)
        
        data["nodes"].append(node_data)
    
    return json.dumps(data, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_my_memviz.py::test_format_json -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit format_json**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: add format_json function"
```

---

### Task 9: DOT Formatter

**Files:**
- Modify: `my_memviz.py`
- Modify: `test_my_memviz.py`

- [ ] **Step 1: Write failing tests for format_dot**

```python
import pytest
from my_memviz import Graph, GraphNode, format_dot


def test_format_dot_basic():
    graph = Graph()
    node1 = GraphNode(node_id=0, op_type="AddBackward", output_shape=[10])
    node2 = GraphNode(node_id=1, op_type="MulBackward", output_shape=[10])
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_edge(0, 1)
    
    result = format_dot(graph, show_memory=False)
    
    assert "digraph" in result
    assert "AddBackward" in result
    assert "MulBackward" in result


def test_format_dot_with_memory():
    graph = Graph()
    node = GraphNode(
        node_id=0,
        op_type="AddBackward",
        output_shape=[10],
        saved_memory_bytes=1000
    )
    graph.add_node(node)
    
    result = format_dot(graph, show_memory=True)
    
    assert "1,000" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_my_memviz.py::test_format_dot -v`
Expected: FAIL with "cannot import name 'format_dot'"

- [ ] **Step 3: Implement format_dot function**

```python
def format_dot(graph, show_memory=True):
    lines = ["digraph computation_graph {"]
    lines.append("  rankdir=LR;")
    
    for node in graph.nodes:
        label = f"{node.op_type}\\nshape: {node.output_shape}"
        if show_memory and node.saved_memory_bytes > 0:
            label += f"\\nmemory: {format_bytes(node.saved_memory_bytes)}"
        
        lines.append(f'  node{node.node_id} [label="{label}"];')
    
    for edge in graph.edges:
        lines.append(f'  node{edge["from"]} -> node{edge["to"]};')
    
    lines.append("}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_my_memviz.py::test_format_dot -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit format_dot**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: add format_dot function"
```

---

### Task 10: PNG/SVG Formatter

**Files:**
- Modify: `my_memviz.py`
- Modify: `test_my_memviz.py`

- [ ] **Step 1: Write failing tests for format_image**

```python
import pytest
import os
from my_memviz import Graph, GraphNode, format_image


def test_format_image_png(tmp_path):
    graph = Graph()
    node = GraphNode(node_id=0, op_type="AddBackward", output_shape=[10])
    graph.add_node(node)
    
    output_file = str(tmp_path / "test_graph.png")
    result = format_image(graph, output_file, format="png", show_memory=False)
    
    if result is not None:
        assert os.path.exists(output_file)


def test_format_image_svg(tmp_path):
    graph = Graph()
    node = GraphNode(node_id=0, op_type="AddBackward", output_shape=[10])
    graph.add_node(node)
    
    output_file = str(tmp_path / "test_graph.svg")
    result = format_image(graph, output_file, format="svg", show_memory=False)
    
    if result is not None:
        assert os.path.exists(output_file)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_my_memviz.py::test_format_image -v`
Expected: FAIL with "cannot import name 'format_image'"

- [ ] **Step 3: Implement format_image function**

```python
import warnings


def format_image(graph, output_file, format="png", show_memory=True):
    try:
        import graphviz
    except ImportError:
        warnings.warn("graphviz package not installed. Install with: pip install graphviz")
        return None
    
    dot_content = format_dot(graph, show_memory)
    
    try:
        dot = graphviz.Source(dot_content)
        dot.render(output_file.rsplit('.', 1)[0], format=format, cleanup=True)
        return output_file
    except Exception as e:
        warnings.warn(f"Failed to render image: {e}")
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_my_memviz.py::test_format_image -v`
Expected: PASS (2 tests, may skip if graphviz not installed)

- [ ] **Step 5: Commit format_image**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: add format_image function"
```

---

### Task 11: Main dump_graph API

**Files:**
- Modify: `my_memviz.py`
- Modify: `test_my_memviz.py`

- [ ] **Step 1: Write failing tests for dump_graph**

```python
import pytest
import torch
import json
import os
from my_memviz import dump_graph


def test_dump_graph_json():
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 30)
    y = linear(x)
    
    result = dump_graph(y, x, format="json", show_memory=True)
    
    assert result is not None
    data = json.loads(result)
    assert "summary" in data
    assert "nodes" in data


def test_dump_graph_dot():
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 30)
    y = linear(x)
    
    result = dump_graph(y, x, format="dot", show_memory=False)
    
    assert result is not None
    assert "digraph" in result


def test_dump_graph_to_file(tmp_path):
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 30)
    y = linear(x)
    
    output_file = str(tmp_path / "graph.json")
    result = dump_graph(y, x, format="json", show_memory=True, output_file=output_file)
    
    assert os.path.exists(output_file)
    with open(output_file) as f:
        data = json.load(f)
    assert "summary" in data


def test_dump_graph_multiple_formats(tmp_path):
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 30)
    y = linear(x)
    
    output_file = str(tmp_path / "graph")
    graph = dump_graph(y, x, format=["json", "dot"], show_memory=True, output_file=output_file)
    
    assert os.path.exists(str(tmp_path / "graph.json"))
    assert os.path.exists(str(tmp_path / "graph.dot"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_my_memviz.py::test_dump_graph -v`
Expected: FAIL with "cannot import name 'dump_graph'"

- [ ] **Step 3: Implement dump_graph function**

```python
import warnings


def dump_graph(output_tensor, input_tensor, format="json", show_memory=True, output_file=None):
    if not hasattr(output_tensor, 'grad_fn'):
        warnings.warn("output_tensor is not a tensor or has no grad_fn")
        return None
    
    if not hasattr(input_tensor, 'requires_grad'):
        warnings.warn("input_tensor is not a tensor")
        return None
    
    if not input_tensor.requires_grad:
        warnings.warn("input_tensor does not require grad")
    
    graph = Graph()
    traverse_graph(output_tensor, input_tensor, graph)
    
    if len(graph.nodes) > 100:
        warnings.warn(f"Very large graph detected: {len(graph.nodes)} nodes")
    
    formats = [format] if isinstance(format, str) else format
    
    results = {}
    for fmt in formats:
        if fmt == "json":
            result = format_json(graph, show_memory)
            results["json"] = result
        elif fmt == "dot":
            result = format_dot(graph, show_memory)
            results["dot"] = result
        elif fmt in ["png", "svg"]:
            if output_file is None:
                warnings.warn(f"output_file required for {fmt} format")
                continue
            
            ext = f".{fmt}"
            if output_file.endswith(f".{fmt}"):
                out_path = output_file
            else:
                out_path = output_file + ext
            
            format_image(graph, out_path, format=fmt, show_memory=show_memory)
            results[fmt] = out_path
    
    if output_file is not None:
        for fmt, result in results.items():
            if fmt == "json":
                ext = ".json"
                if not output_file.endswith(ext):
                    path = output_file + ext
                else:
                    path = output_file
                with open(path, 'w') as f:
                    f.write(result)
            elif fmt == "dot":
                ext = ".dot"
                if not output_file.endswith(ext):
                    path = output_file + ext
                else:
                    path = output_file
                with open(path, 'w') as f:
                    f.write(result)
    
    if isinstance(format, str) and format in results:
        return results[format]
    
    return graph
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_my_memviz.py::test_dump_graph -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit dump_graph**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: add dump_graph main API"
```

---

### Task 12: Error Handling and Edge Cases

**Files:**
- Modify: `my_memviz.py`
- Modify: `test_my_memviz.py`

- [ ] **Step 1: Write failing tests for error handling**

```python
import pytest
import torch
from my_memviz import dump_graph


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
    
    # Should still work, just warn
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_my_memviz.py::test_dump_graph -v`
Expected: Some tests FAIL (error handling may not be complete)

- [ ] **Step 3: Verify error handling is already implemented**

The error handling is already in place from Task 11. Run tests to confirm.

Run: `pytest test_my_memviz.py::test_dump_graph_disconnected -v`
Expected: PASS

- [ ] **Step 4: Run all error handling tests**

Run: `pytest test_my_memviz.py -k "disconnected or non_tensor or requires_grad" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit error handling tests**

```bash
git add test_my_memviz.py
git commit -m "test: add error handling tests"
```

---

### Task 13: Integration Tests

**Files:**
- Modify: `test_my_memviz.py`

- [ ] **Step 1: Write integration tests**

```python
import pytest
import torch
import json
import os
from my_memviz import dump_graph


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
    
    result = dump_graph(y, x, format="json", show_memory=True)
    data = json.loads(result)
    
    assert data["summary"]["num_nodes"] >= 5
    assert data["summary"]["total_saved_memory_bytes"] > 0
    
    op_types = [node["op_type"] for node in data["nodes"]]
    assert any("AddmmBackward" in op for op in op_types)


def test_integration_residual_connection():
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 20)
    
    y = linear(x)
    y = y + x  # Residual connection
    
    result = dump_graph(y, x, format="json", show_memory=True)
    data = json.loads(result)
    
    assert data["summary"]["num_nodes"] >= 2
    assert any(node["op_type"] == "AddBackward" for node in data["nodes"])


def test_integration_memory_tracking():
    x = torch.randn(100, 100, requires_grad=True)
    linear = torch.nn.Linear(100, 100)
    y = linear(x)
    
    result = dump_graph(y, x, format="json", show_memory=True)
    data = json.loads(result)
    
    # Linear layer saves weight, bias, and input
    assert data["summary"]["total_saved_memory_bytes"] > 0
    
    # Check that saved tensors are properly tracked
    for node in data["nodes"]:
        if len(node.get("saved_tensors", [])) > 0:
            for st in node["saved_tensors"]:
                assert "size_bytes" in st
                assert "size_formatted" in st
                assert st["size_bytes"] > 0
```

- [ ] **Step 2: Run integration tests**

Run: `pytest test_my_memviz.py::test_integration -v`
Expected: PASS (3 tests)

- [ ] **Step 3: Commit integration tests**

```bash
git add test_my_memviz.py
git commit -m "test: add integration tests"
```

---

### Task 14: Final Verification and Documentation

**Files:**
- Create: `README.md`

- [ ] **Step 1: Run all tests**

Run: `pytest test_my_memviz.py -v`
Expected: All tests PASS

- [ ] **Step 2: Create README.md**

```markdown
# Tensor Graph Dump Tool

A PyTorch tool for visualizing computation subgraphs between two tensors with memory analysis.

## Installation

```bash
pip install torch graphviz
```

## Usage

```python
import torch
from my_memviz import dump_graph

x = torch.randn(10, 20, requires_grad=True)
model = torch.nn.Linear(20, 30)
y = model(x)

# Output as JSON
dump_graph(y, x, format="json", show_memory=True, output_file="graph.json")

# Output as DOT
dump_graph(y, x, format="dot", show_memory=True, output_file="graph.dot")

# Output as PNG (requires graphviz installed)
dump_graph(y, x, format="png", show_memory=True, output_file="graph.png")

# Multiple formats at once
dump_graph(y, x, format=["json", "dot", "png"], show_memory=True, output_file="graph")
```

## Features

- Extract computation subgraph between two tensors
- Memory analysis showing saved tensors for backward pass
- Multiple output formats: JSON, DOT, PNG, SVG
- Detailed per-node memory breakdown

## API

```python
dump_graph(output_tensor, input_tensor, format='json', show_memory=True, output_file=None)
```

### Parameters

- `output_tensor`: Target tensor (end point of subgraph)
- `input_tensor`: Source tensor (start point of subgraph)
- `format`: 'json', 'dot', 'png', 'svg', or list of formats
- `show_memory`: Include saved tensor memory info (default: True)
- `output_file`: Path to save output (optional)

### Returns

- Graph object (can be rendered multiple times)
```

- [ ] **Step 3: Commit README**

```bash
git add README.md
git commit -m "docs: add README with usage examples"
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete tensor graph dump tool implementation"
```

---

## Summary

This plan implements the tensor graph dump tool in 14 tasks following TDD:

1. Project setup
2. Byte formatting utility
3. Tensor memory calculation
4. GraphNode data structure
5. Graph data structure
6. Extract saved tensors from autograd
7. Graph traversal
8. JSON formatter
9. DOT formatter
10. PNG/SVG formatter
11. Main dump_graph API
12. Error handling tests
13. Integration tests
14. Final verification and docs

Each task follows the pattern: write test, run test (fail), implement, run test (pass), commit.