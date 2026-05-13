# dump_graph Context Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert dump_graph to a context manager that automatically captures tensor operations and call stacks within a with block.

**Architecture:** Monkey patch torch.Tensor.__new__ to track tensor creation during with block, then reconstruct subgraph from end nodes on exit.

**Tech Stack:** Python, PyTorch, contextlib, traceback

---

## File Structure

**Modified Files:**
- `my_memviz.py` - Main implementation file
- `test_my_memviz.py` - All tests updated for new API

**Key Changes:**
- `GraphNode` dataclass: add `call_stack` field
- `dump_graph`: rewritten as `@contextmanager`
- New internal functions: `_track_tensor_creation`, `_find_end_nodes`, `_build_subgraph`
- `traverse_graph`: removed from public API, logic internalized

---

### Task 1: Add call_stack to GraphNode

**Files:**
- Modify: `my_memviz.py:40-47`
- Test: `test_my_memviz.py`

- [ ] **Step 1: Write failing test for call_stack field**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_my_memviz.py::test_graph_node_with_call_stack -v`
Expected: FAIL - "GraphNode() got an unexpected keyword argument 'call_stack'"

- [ ] **Step 3: Add call_stack field to GraphNode**

```python
@dataclass
class GraphNode:
    node_id: int
    op_type: str
    output_shape: List[int]
    saved_tensors: List[Dict[str, Any]] = field(default_factory=list)
    saved_memory_bytes: int = 0
    call_stack: List[Dict[str, Any]] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_my_memviz.py::test_graph_node_with_call_stack -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: add call_stack field to GraphNode"
```

---

### Task 2: Update format_json to include call_stack

**Files:**
- Modify: `my_memviz.py:136-163`
- Test: `test_my_memviz.py`

- [ ] **Step 1: Write failing test for call_stack in JSON output**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_my_memviz.py::test_format_json_with_call_stack -v`
Expected: FAIL - "call_stack not in data['nodes'][0]"

- [ ] **Step 3: Update format_json to include call_stack**

```python
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
            "output_shape": node.output_shape,
            "call_stack": node.call_stack
        }
        
        if show_memory:
            node_data["saved_tensors"] = node.saved_tensors
            node_data["saved_memory_bytes"] = node.saved_memory_bytes
            node_data["saved_memory_formatted"] = format_bytes(node.saved_memory_bytes)
        
        data["nodes"].append(node_data)
    
    return json.dumps(data, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_my_memviz.py::test_format_json_with_call_stack -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: include call_stack in JSON output"
```

---

### Task 3: Update format_dot to include call_stack info

**Files:**
- Modify: `my_memviz.py:166-181`
- Test: `test_my_memviz.py`

- [ ] **Step 1: Write test for call_stack in DOT output**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_my_memviz.py::test_format_dot_with_call_stack -v`
Expected: FAIL - "model.py:15 not in result"

- [ ] **Step 3: Update format_dot to include abbreviated call_stack**

```python
def format_dot(graph, show_memory=True):
    lines = ["digraph computation_graph {"]
    lines.append("  rankdir=LR;")
    
    for node in graph.nodes:
        label = f"{node.op_type}\\nshape: {node.output_shape}"
        if show_memory and node.saved_memory_bytes > 0:
            label += f"\\nmemory: {format_bytes(node.saved_memory_bytes)}"
        
        if node.call_stack:
            first_call = node.call_stack[0]
            label += f"\\n{first_call['file']}:{first_call['line']} in {first_call['function']}"
        
        lines.append(f'  node{node.node_id} [label="{label}"];')
    
    for edge in graph.edges:
        lines.append(f'  node{edge["from"]} -> node{edge["to"]};')
    
    lines.append("}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_my_memviz.py::test_format_dot_with_call_stack -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: include call_stack info in DOT output"
```

---

### Task 4: Create _extract_call_stack helper function

**Files:**
- Modify: `my_memviz.py`
- Test: `test_my_memviz.py`

- [ ] **Step 1: Write test for call stack extraction**

```python
def test_extract_call_stack():
    import traceback
    
    stack = traceback.extract_stack()
    result = _extract_call_stack(stack)
    
    assert len(result) > 0
    assert "file" in result[0]
    assert "line" in result[0]
    assert "function" in result[0]
    assert "code" in result[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_my_memviz.py::test_extract_call_stack -v`
Expected: FAIL - "NameError: name '_extract_call_stack' is not defined"

- [ ] **Step 3: Add _extract_call_stack function**

Add at line 38 (after `extract_saved_tensors`):

```python
def _extract_call_stack(stack_summary):
    """
    Extract call stack information from traceback.extract_stack() result.
    
    Returns list of dicts with file, line, function, code keys.
    """
    call_stack = []
    for frame in stack_summary:
        call_stack.append({
            "file": frame.filename,
            "line": frame.lineno,
            "function": frame.name,
            "code": frame.line if frame.line else ""
        })
    return call_stack
```

Update imports at top of file:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any
import warnings
import json
import traceback
import os
from contextlib import contextmanager
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_my_memviz.py::test_extract_call_stack -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: add _extract_call_stack helper function"
```

---

### Task 5: Create _track_tensor_creation function

**Files:**
- Modify: `my_memviz.py`
- Test: `test_my_memviz.py`

- [ ] **Step 1: Write test for tensor tracking**

```python
def test_track_tensor_creation():
    captured = []
    original_new = torch.Tensor.__new__
    
    def tracked_new(cls, *args, **kwargs):
        tensor = original_new(cls)
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            tensor = args[0].clone()
        else:
            tensor = torch.empty(0)
        tensor.requires_grad = kwargs.get('requires_grad', False)
        _track_tensor_creation(tensor, captured)
        return tensor
    
    torch.Tensor.__new__ = tracked_new
    
    try:
        x = torch.randn(10, 20, requires_grad=True)
        y = torch.randn(10, 20, requires_grad=False)
        
        assert len(captured) >= 1
        assert any(t['requires_grad'] for t in captured)
    finally:
        torch.Tensor.__new__ = original_new
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_my_memviz.py::test_track_tensor_creation -v`
Expected: FAIL - "NameError: name '_track_tensor_creation' is not defined"

- [ ] **Step 3: Add _track_tensor_creation function**

Add after `_extract_call_stack`:

```python
def _track_tensor_creation(tensor, captured_list):
    """
    Track a newly created tensor and its metadata.
    
    Args:
        tensor: The newly created tensor
        captured_list: List to append tensor info to
    """
    stack = traceback.extract_stack()
    call_stack = _extract_call_stack(stack)
    
    tensor_info = {
        "id": id(tensor),
        "tensor": tensor,
        "grad_fn": tensor.grad_fn,
        "requires_grad": tensor.requires_grad,
        "shape": list(tensor.shape) if hasattr(tensor, 'shape') else [],
        "dtype": str(tensor.dtype).replace("torch.", "") if hasattr(tensor, 'dtype') else "",
        "call_stack": call_stack
    }
    captured_list.append(tensor_info)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_my_memviz.py::test_track_tensor_creation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: add _track_tensor_creation function"
```

---

### Task 6: Create _find_end_nodes function

**Files:**
- Modify: `my_memviz.py`
- Test: `test_my_memviz.py`

- [ ] **Step 1: Write test for finding end nodes**

```python
def test_find_end_nodes():
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 30)
    y = linear(x)
    z = y * 2
    
    captured = []
    _track_tensor_creation(y, captured)
    _track_tensor_creation(z, captured)
    
    end_nodes = _find_end_nodes(captured)
    
    assert len(end_nodes) >= 1
    assert any(t['tensor'] is z for t in end_nodes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_my_memviz.py::test_find_end_nodes -v`
Expected: FAIL - "NameError: name '_find_end_nodes' is not defined"

- [ ] **Step 3: Add _find_end_nodes function**

Add after `_track_tensor_creation`:

```python
def _find_end_nodes(captured_tensors):
    """
    Find tensors that are not dependencies of other captured tensors.
    
    An end node is a tensor whose grad_fn is not referenced by any other
    captured tensor's grad_fn.next_functions.
    
    Args:
        captured_tensors: List of captured tensor info dicts
        
    Returns:
        List of captured tensor info dicts that are end nodes
    """
    if not captured_tensors:
        return []
    
    all_grad_fn_ids = set()
    for t_info in captured_tensors:
        if t_info['grad_fn'] is not None:
            all_grad_fn_ids.add(id(t_info['grad_fn']))
    
    referenced_grad_fn_ids = set()
    for t_info in captured_tensors:
        if t_info['grad_fn'] is not None:
            for next_fn, _ in t_info['grad_fn'].next_functions:
                if next_fn is not None:
                    referenced_grad_fn_ids.add(id(next_fn))
    
    end_nodes = []
    for t_info in captured_tensors:
        if t_info['grad_fn'] is not None:
            if id(t_info['grad_fn']) not in referenced_grad_fn_ids:
                end_nodes.append(t_info)
    
    if not end_nodes:
        for t_info in captured_tensors:
            if t_info['grad_fn'] is not None:
                end_nodes.append(t_info)
    
    return end_nodes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_my_memviz.py::test_find_end_nodes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: add _find_end_nodes function"
```

---

### Task 7: Create _build_subgraph function

**Files:**
- Modify: `my_memviz.py`
- Test: `test_my_memviz.py`

- [ ] **Step 1: Write test for building subgraph**

```python
def test_build_subgraph():
    x = torch.randn(10, 20, requires_grad=True)
    linear = torch.nn.Linear(20, 30)
    y = linear(x)
    
    captured = []
    _track_tensor_creation(y, captured)
    
    captured_ids = {t['id'] for t in captured}
    end_nodes = _find_end_nodes(captured)
    
    graph = _build_subgraph(end_nodes, captured_ids)
    
    assert isinstance(graph, Graph)
    assert len(graph.nodes) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_my_memviz.py::test_build_subgraph -v`
Expected: FAIL - "NameError: name '_build_subgraph' is not defined"

- [ ] **Step 3: Add _build_subgraph function**

Add after `_find_end_nodes`:

```python
def _build_subgraph(end_nodes, captured_tensor_ids):
    """
    Build a Graph from end nodes by traversing grad_fn.next_functions.
    
    Only includes nodes that were captured during the with block.
    
    Args:
        end_nodes: List of end node tensor info dicts
        captured_tensor_ids: Set of tensor IDs captured during with block
        
    Returns:
        Graph object with nodes and edges
    """
    graph = Graph()
    visited_grad_fns = {}
    grad_fn_to_tensor_info = {}
    
    for t_info in end_nodes:
        if t_info['grad_fn'] is not None:
            grad_fn_to_tensor_info[id(t_info['grad_fn'])] = t_info
    
    def traverse(grad_fn, parent_id=None):
        if grad_fn is None or id(grad_fn) in visited_grad_fns:
            return visited_grad_fns.get(id(grad_fn))
        
        node_id = len(graph.nodes)
        visited_grad_fns[id(grad_fn)] = node_id
        
        op_type = type(grad_fn).__name__
        
        output_shape = []
        t_info = grad_fn_to_tensor_info.get(id(grad_fn))
        if t_info:
            output_shape = t_info['shape']
        
        saved_tensors = extract_saved_tensors(grad_fn)
        saved_memory = sum(st["size_bytes"] for st in saved_tensors)
        
        call_stack = []
        if t_info:
            call_stack = t_info['call_stack']
        
        node = GraphNode(
            node_id=node_id,
            op_type=op_type,
            output_shape=output_shape,
            saved_tensors=saved_tensors,
            saved_memory_bytes=saved_memory,
            call_stack=call_stack
        )
        graph.add_node(node)
        
        for next_fn, _ in grad_fn.next_functions:
            if next_fn is not None:
                child_id = traverse(next_fn, node_id)
                if child_id is not None and child_id != node_id:
                    graph.add_edge(node_id, child_id)
        
        return node_id
    
    for end_node in end_nodes:
        if end_node['grad_fn'] is not None:
            traverse(end_node['grad_fn'])
    
    return graph
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_my_memviz.py::test_build_subgraph -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: add _build_subgraph function"
```

---

### Task 8: Rewrite dump_graph as context manager

**Files:**
- Modify: `my_memviz.py:202-246`
- Test: `test_my_memviz.py`

- [ ] **Step 1: Write test for basic context manager usage**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_my_memviz.py::test_dump_graph_context_manager_basic -v`
Expected: FAIL - TypeError about dump_graph signature

- [ ] **Step 3: Rewrite dump_graph as context manager**

Replace `dump_graph` function (lines 202-246):

```python
@contextmanager
def dump_graph(format=["json", "svg"], show_memory=True, output_file=None):
    """
    Context manager to capture computation graph within a with block.
    
    Args:
        format: Output format(s). Default ["json", "svg"]. Accepts string or list.
        show_memory: Include memory information. Default True.
        output_file: Output file path (without extension). Default None (auto-generates dump_{pid}).
    
    Usage:
        with dump_graph():
            x = torch.randn(10, 20, requires_grad=True)
            y = model(x)
        # Saves dump_{pid}.json and dump_{pid}.svg
    """
    captured_tensors = []
    original_new = torch.Tensor.__new__
    
    def tracked_new(cls, *args, **kwargs):
        tensor = original_new(cls)
        if 'device' in kwargs:
            pass
        if 'requires_grad' in kwargs:
            tensor.requires_grad_(kwargs['requires_grad'])
        _track_tensor_creation(tensor, captured_tensors)
        return tensor
    
    torch.Tensor.__new__ = tracked_new
    
    try:
        yield None
    finally:
        torch.Tensor.__new__ = original_new
        
        end_nodes = _find_end_nodes(captured_tensors)
        
        if not end_nodes:
            warnings.warn("No computation graph captured in with block")
            graph = Graph()
        else:
            captured_ids = {t['id'] for t in captured_tensors}
            graph = _build_subgraph(end_nodes, captured_ids)
        
        if output_file is None:
            import sys
            pid = os.getpid()
            base_name = f"dump_{pid}"
        else:
            base_name = output_file
        
        formats = [format] if isinstance(format, str) else format
        
        for fmt in formats:
            if fmt in ["png", "svg"]:
                ext = f".{fmt}"
                out_path = base_name if base_name.endswith(ext) else base_name + ext
                graph.render(fmt, show_memory=show_memory, output_file=out_path)
            else:
                result = graph.render(fmt, show_memory=show_memory)
                if result is not None:
                    ext = f".{fmt}"
                    out_path = base_name if base_name.endswith(ext) else base_name + ext
                    with open(out_path, 'w') as f:
                        f.write(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_my_memviz.py::test_dump_graph_context_manager_basic -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add my_memviz.py test_my_memviz.py
git commit -m "feat: rewrite dump_graph as context manager"
```

---

### Task 9: Add test for default file naming with PID

**Files:**
- Test: `test_my_memviz.py`

- [ ] **Step 1: Write test for default PID-based file naming**

```python
def test_dump_graph_default_file_naming():
    import os
    pid = os.getpid()
    
    with dump_graph(format="json"):
        x = torch.randn(10, 20, requires_grad=True)
        y = x + 1
    
    expected_file = f"dump_{pid}.json"
    assert os.path.exists(expected_file)
    
    with open(expected_file, "r") as f:
        data = json.loads(f.read())
    assert data["summary"]["num_nodes"] >= 1
    
    os.remove(expected_file)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest test_my_memviz.py::test_dump_graph_default_file_naming -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add test_my_memviz.py
git commit -m "test: add test for default PID-based file naming"
```

---

### Task 10: Add test for multiple formats

**Files:**
- Test: `test_my_memviz.py`

- [ ] **Step 1: Write test for multiple output formats**

```python
def test_dump_graph_multiple_formats():
    with dump_graph(format=["json", "dot"], output_file="test_multi"):
        x = torch.randn(10, 20, requires_grad=True)
        y = x * 2
    
    assert os.path.exists("test_multi.json")
    assert os.path.exists("test_multi.dot")
    
    os.remove("test_multi.json")
    os.remove("test_multi.dot")
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest test_my_memviz.py::test_dump_graph_multiple_formats -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add test_my_memviz.py
git commit -m "test: add test for multiple output formats"
```

---

### Task 11: Add test for empty with block warning

**Files:**
- Test: `test_my_memviz.py`

- [ ] **Step 1: Write test for empty with block warning**

```python
def test_dump_graph_empty_block():
    import os
    pid = os.getpid()
    
    with pytest.warns(UserWarning):
        with dump_graph(format="json"):
            pass
    
    expected_file = f"dump_{pid}.json"
    if os.path.exists(expected_file):
        with open(expected_file, "r") as f:
            data = json.loads(f.read())
        assert data["summary"]["num_nodes"] == 0
        os.remove(expected_file)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest test_my_memviz.py::test_dump_graph_empty_block -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add test_my_memviz.py
git commit -m "test: add test for empty with block warning"
```

---

### Task 12: Add integration test for multi-layer network

**Files:**
- Test: `test_my_memviz.py`

- [ ] **Step 1: Write integration test for multi-layer network**

```python
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
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest test_my_memviz.py::test_integration_context_manager_multi_layer -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add test_my_memviz.py
git commit -m "test: add integration test for multi-layer network with context manager"
```

---

### Task 13: Remove old API tests and update imports

**Files:**
- Modify: `test_my_memviz.py:1-6`

- [ ] **Step 1: Remove traverse_graph from imports**

Current import line 5:
```python
from my_memviz import format_bytes, calculate_tensor_memory, GraphNode, Graph, extract_saved_tensors, traverse_graph, format_json, format_dot, format_image, dump_graph
```

Replace with:
```python
from my_memviz import format_bytes, calculate_tensor_memory, GraphNode, Graph, extract_saved_tensors, format_json, format_dot, format_image, dump_graph, _extract_call_stack, _track_tensor_creation, _find_end_nodes, _build_subgraph
```

- [ ] **Step 2: Run all tests to check for failures**

Run: `pytest test_my_memviz.py -v`
Expected: Some tests fail due to old API usage

- [ ] **Step 3: Remove/update tests that used old dump_graph API**

Delete or update the following tests that used the old `dump_graph(output_tensor, input_tensor)` API:
- `test_dump_graph_json` (line 276-286)
- `test_dump_graph_dot` (line 289-298)
- `test_dump_graph_to_file` (line 301-312)
- `test_dump_graph_multiple_formats` (line 315-325)
- `test_dump_graph_disconnected_tensors` (line 328-335)
- `test_dump_graph_non_tensor_input` (line 338-344)
- `test_dump_graph_no_requires_grad` (line 347-352)
- `test_integration_multi_layer_network` (line 355-370)
- `test_integration_residual_connection` (line 373-384)
- `test_integration_memory_tracking` (line 387-402)

These tests are replaced by the new context manager tests in Tasks 9-12.

- [ ] **Step 4: Remove traverse_graph tests**

Delete tests:
- `test_traverse_graph_simple` (line 126-135)
- `test_traverse_graph_multi_layer` (line 138-151)
- `test_traverse_graph_no_path` (line 154-161)

These functions are now internal and tested indirectly through the context manager tests.

- [ ] **Step 5: Run all tests to verify they pass**

Run: `pytest test_my_memviz.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add test_my_memviz.py
git commit -m "refactor: remove old API tests and update imports"
```

---

### Task 14: Remove public traverse_graph and clean up my_memviz.py

**Files:**
- Modify: `my_memviz.py:75-134`

- [ ] **Step 1: Remove traverse_graph public function**

The `traverse_graph` function (lines 75-134) is no longer needed as public API. Its logic is now in `_build_subgraph`. Delete the entire function.

- [ ] **Step 2: Run tests to verify nothing breaks**

Run: `pytest test_my_memviz.py -v`
Expected: All tests PASS (traverse_graph tests already removed in Task 13)

- [ ] **Step 3: Commit**

```bash
git add my_memviz.py
git commit -m "refactor: remove public traverse_graph API"
```

---

### Task 15: Final verification and cleanup

**Files:**
- All files

- [ ] **Step 1: Run complete test suite**

Run: `pytest test_my_memviz.py -v`
Expected: All tests PASS

- [ ] **Step 2: Check for any leftover dump files**

Run: `ls dump_*` or manual check
Expected: No leftover test files (tests should clean up)

- [ ] **Step 3: Verify implementation matches spec**

Read through `my_memviz.py` and verify:
- `GraphNode` has `call_stack` field
- `dump_graph` is `@contextmanager`
- `_extract_call_stack`, `_track_tensor_creation`, `_find_end_nodes`, `_build_subgraph` exist
- Default format is `["json", "svg"]`
- PID-based default file naming works

- [ ] **Step 4: Commit any final fixes**

```bash
git status
git add -A
git commit -m "chore: final cleanup and verification"
```