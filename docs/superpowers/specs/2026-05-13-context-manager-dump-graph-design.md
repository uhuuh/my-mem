# dump_graph Context Manager Design

## Summary

Redesign `dump_graph` as a context manager that automatically captures all tensor operations within a `with` block, eliminating the need to manually specify input/output tensors. Each node includes complete Python call stack information for debugging.

## Motivation

The current `dump_graph(output_tensor, input_tensor, ...)` API requires users to:
1. Know which tensors are inputs and outputs
2. Manually pass them after computation completes
3. Does not capture stack traces for debugging

The context manager approach simplifies usage and adds debugging context.

## API

### New API

```python
@contextmanager
def dump_graph(format=["json", "svg"], show_memory=True, output_file=None):
    """
    Context manager to capture computation graph within a with block.

    Usage:
        with dump_graph():
            x = torch.randn(10, 20, requires_grad=True)
            y = model(x)
        # Automatically saves dump_{pid}.json and dump_{pid}.svg

        with dump_graph(format="json", output_file="graph"):
            x = torch.randn(10, 20, requires_grad=True)
            y = model(x)
        # Saves graph.json
    """
```

### Parameters

- `format`: Output format(s). Default `["json", "svg"]`. Accepts single string or list.
- `show_memory`: Include memory information in output. Default `True`.
- `output_file`: Output file path (without extension). Default `None` (auto-generates `dump_{pid}`).

### Removed API

The original `dump_graph(output_tensor, input_tensor, ...)` function is removed.

## Implementation

### Entry (`__enter__`)

1. Save reference to original `torch.Tensor.__new__`
2. Replace with tracked version that:
   - Calls original `__new__` to create tensor
   - Records tensor metadata: `id`, `grad_fn`, `requires_grad`, `shape`, `dtype`
   - Captures full call stack using `traceback.extract_stack()`
   - Stores in internal `captured_tensors` list
3. Return `None` (context manager yields nothing)

### Exit (`__exit__`)

1. Restore original `torch.Tensor.__new__`
2. Identify end nodes:
   - Tensors with `grad_fn` (computed values)
   - Not referenced by any other captured tensor's `grad_fn.next_functions`
3. Build subgraph:
   - Start from end nodes
   - Traverse `grad_fn.next_functions` recursively
   - Only include nodes captured during the with block
4. For each node, extract:
   - Operation type from `grad_fn`
   - Output shape from tensor
   - Saved tensors and memory
   - Call stack recorded at creation time
5. Output to files:
   - If `output_file` is `None`: use `dump_{pid}` as base name
   - For each format in `format` list: render and save with appropriate extension

### Data Structures

#### GraphNode (Updated)

```python
@dataclass
class GraphNode:
    node_id: int
    op_type: str
    output_shape: List[int]
    saved_tensors: List[Dict[str, Any]] = field(default_factory=list)
    saved_memory_bytes: int = 0
    call_stack: List[Dict[str, Any]] = field(default_factory=list)  # NEW
```

#### Call Stack Entry

```json
{
  "file": "/path/to/file.py",
  "line": 42,
  "function": "forward",
  "code": "y = self.linear(x)"
}
```

### Internal Functions

- `_track_tensor_creation(tensor, captured_tensors, call_stack_cache)`: Hook for `__new__`, stores tensor info
- `_find_end_nodes(captured_tensors)`: Returns list of tensors that are not dependencies of others
- `_build_subgraph(end_nodes, captured_tensor_ids)`: Traverses and builds graph from end nodes

## Output Behavior

### File Naming

| `output_file` | `format` | Result |
|---------------|----------|--------|
| `None` | `["json", "svg"]` | `dump_12345.json`, `dump_12345.svg` |
| `None` | `"json"` | `dump_12345.json` |
| `"graph"` | `["json", "svg"]` | `graph.json`, `graph.svg` |
| `"my_output"` | `"dot"` | `my_output.dot` |
| `"output.json"` | `"json"` | `output.json` (no double extension) |

### Default Behavior

```python
with dump_graph():  # Uses default format=["json", "svg"]
    x = torch.randn(10, 20, requires_grad=True)
    y = model(x)
# Creates: dump_{pid}.json and dump_{pid}.svg
```

## Example Output

```json
{
  "summary": {
    "num_nodes": 3,
    "total_saved_memory_bytes": 8400,
    "total_saved_memory_formatted": "8,400"
  },
  "nodes": [
    {
      "id": 0,
      "op_type": "AddmmBackward",
      "output_shape": [10, 30],
      "saved_tensors": [...],
      "saved_memory_bytes": 5600,
      "call_stack": [
        {
          "file": "/home/user/model.py",
          "line": 15,
          "function": "forward",
          "code": "y = self.linear(x)"
        }
      ]
    }
  ],
  "edges": [...]
}
```

## Edge Cases

### No Computation in Block

If no tensors with `grad_fn` are created (no operations performed), the context manager:
- Still creates output files
- Reports 0 nodes in summary
- Issues a warning: `"No computation graph captured in with block"`

### Exception During Block

If an exception occurs in the with block:
- `__exit__` still executes and attempts to capture any tensors created before the exception
- Original exception propagates normally
- Output files are still generated with partial graph

### Nested Context Managers

Not supported. If `dump_graph` is nested:
- Inner context manager takes over tracking
- Outer context manager sees no new tensors
- Warning issued on nested entry

## Testing Strategy

1. **Unit Tests**:
   - Tensor capture tracking
   - End node identification
   - Call stack extraction
   - File naming logic

2. **Integration Tests**:
   - Single layer network
   - Multi-layer network
   - Residual connections
   - Multiple outputs
   - Different formats

3. **Edge Cases**:
   - Empty with block
   - No gradient operations
   - Nested context managers
   - Exception during with block

## Migration

This is a breaking change. Users must update from:

```python
# Old
x = torch.randn(10, 20, requires_grad=True)
y = model(x)
dump_graph(y, x, format="json", output_file="graph")
```

To:

```python
# New
with dump_graph(format="json", output_file="graph"):
    x = torch.randn(10, 20, requires_grad=True)
    y = model(x)
```

## File Changes

- `my_memviz.py`:
  - Rewrite `dump_graph` as context manager
  - Add `call_stack` to `GraphNode`
  - Add `_track_tensor_creation()`
  - Add `_find_end_nodes()`
  - Add `_build_subgraph()`
  - Remove public `traverse_graph` (logic internalized)

- `test_my_memviz.py`:
  - Update all tests for new API
  - Add tests for context manager behavior
  - Add tests for call stack capture
  - Add tests for file naming