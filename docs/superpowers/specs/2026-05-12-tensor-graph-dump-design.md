# Tensor Graph Dump Tool Design

## Overview

A PyTorch tool similar to torchviz that dumps the computation subgraph between two tensors, with support for text (JSON), DOT, and PNG/SVG formats. Includes optional memory analysis showing saved tensor sizes for backward pass - primarily for performance analysis to identify memory bottlenecks.

## Core API

**Main entry point:**
```python
dump_graph(output_tensor, input_tensor, format='json', show_memory=True, output_file=None)
```

**Parameters:**
- `output_tensor`: The target tensor (end point of subgraph)
- `input_tensor`: The source tensor (start point of subgraph)
- `format`: 'json', 'dot', or 'png' (or list of formats)
- `show_memory`: Whether to include saved tensor memory info (shows both summary and detailed breakdown)
- `output_file`: Path to save output (auto-generates from format if None)

**Returns:** Graph object that can be rendered multiple times

**Example usage:**
```python
x = torch.randn(10, requires_grad=True)
y = model(x)
dump_graph(y, x, format=['json', 'png'], show_memory=True)
```

## Graph Traversal & Node Structure

**Graph traversal:**
- Start from `output_tensor.grad_fn` (the backward function)
- Follow the `next_functions` chain recursively
- Stop when reaching a node that depends on `input_tensor` (where `input_tensor` is in the node's inputs or saved tensors)
- Collect all nodes on all paths from output to input

**Node information captured:**
- Operation type (e.g., 'AddmmBackward', 'ReluBackward')
- Input/output tensor shapes
- Saved tensors metadata per node (shapes, dtypes, sizes)

**Memory reporting (when `show_memory=True`):**
- **Summary**: Total saved tensor memory across entire subgraph (both formatted string with thousand separators and raw numeric value)
- **Detailed**: Per-node breakdown showing each saved tensor's name, shape, dtype, size with both formatted string and raw numeric value

**Graph structure:**
- Directed acyclic graph (DAG) representation
- Nodes: operations with metadata
- Edges: data flow between operations

## Output Formats

**JSON format:**
- Structured JSON representing the graph
- Example:
```json
{
  "summary": {
    "total_saved_memory": "129,400,000",
    "total_saved_memory_bytes": 129400000,
    "num_nodes": 3
  },
  "nodes": [
    {
      "id": 0,
      "op_type": "AddmmBackward",
      "output_shape": [100, 50],
      "saved_tensors": [
        {"name": "mat1", "shape": [100, 200], "dtype": "float32", "size": "80,000,000", "size_bytes": 80000000},
        {"name": "bias", "shape": [50], "dtype": "float32", "size": "200", "size_bytes": 200}
      ],
      "saved_memory": "80,000,200",
      "saved_memory_bytes": 80000200
    }
  ],
  "edges": [
    {"from": 0, "to": 1},
    {"from": 0, "to": 2}
  ]
}
```

**DOT format:**
- Graphviz DOT syntax
- Nodes labeled with operation type and shapes
- Memory info in node labels with formatted byte strings

**PNG/SVG format:**
- Render DOT using graphviz backend
- Memory info displayed in node labels

## Architecture

**Single file structure:**
```
my_memviz.py    # All implementation in one file
```

**Components in one file:**

1. **GraphBuilder** - Traverses autograd graph from output to input, collects nodes/edges, extracts saved tensor metadata

2. **MemoryAnalyzer** - Calculates tensor memory sizes, formats bytes with thousand separators

3. **Graph class** - Holds nodes/edges data, `render(format)` method, `save(path, format)` method

4. **Formatter functions** - JSON, DOT, and image formatters as internal functions

5. **Main API** - `dump_graph()` function as entry point

**Dependencies:**
- `torch` (required)
- `graphviz` Python package (optional, only for PNG/SVG output)

## Error Handling & Edge Cases

**Error handling (all via warnings, no exceptions):**
- `output_tensor` or `input_tensor` is not a tensor → print warning, return None
- Tensors not connected in computation graph → print warning, return minimal/empty graph
- `format='png'` but graphviz not installed → print warning with installation instructions, skip PNG output
- `input_tensor` requires_grad=False → print warning, attempt to proceed

**Edge cases:**
- Multiple paths between tensors → include all nodes in the subgraph
- Cycles in graph → print warning, still include nodes
- Empty graph (tensors are the same) → return minimal graph with one node
- Very large graphs → print warning if graph is unusually deep (>100 nodes)

**Validation:**
- Check tensors are connected before traversal, warn if not
- All error conditions print warnings but try to continue gracefully

## Testing Approach

**Test categories:**

1. **Basic functionality tests:**
   - Simple linear layer forward pass
   - Multi-layer network with ReLU activations
   - Residual connections (multiple paths)

2. **Graph traversal tests:**
   - Single path between tensors
   - Multiple paths between tensors
   - Disconnected tensors (warning case)
   - Same tensor as input and output

3. **Memory tracking tests:**
   - Verify saved tensor count matches expected
   - Verify memory calculation accuracy
   - Test with different dtypes (float32, float16, etc.)

4. **Output format tests:**
   - JSON output validity
   - DOT syntax correctness
   - PNG/SVG generation (when graphviz available)

5. **Edge case tests:**
   - Very deep graphs
   - Graphs with no saved tensors
   - Non-leaf input tensors

**Test framework:** pytest (standard for Python projects)