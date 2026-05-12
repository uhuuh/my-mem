# my-memviz

A PyTorch computation graph visualization tool that tracks saved tensors and memory usage.

## Installation

```bash
pip install torch
pip install graphviz  # Optional, for PNG/SVG output
```

## Quick Start

```python
import torch
from my_memviz import dump_graph

# Create a simple computation graph
x = torch.randn(10, 20, requires_grad=True)
linear = torch.nn.Linear(20, 30)
y = linear(x)

# Export to JSON
result = dump_graph(y, x, format="json")
print(result)

# Export to DOT format
result = dump_graph(y, x, format="dot")

# Export to file
dump_graph(y, x, format="json", output_file="graph")
```

## API Reference

### `dump_graph(output_tensor, input_tensor, format="json", show_memory=True, output_file=None)`

Main function to dump the computation graph between two tensors.

**Parameters:**
- `output_tensor`: The output tensor of the computation graph
- `input_tensor`: The input tensor (must have `requires_grad=True`)
- `format`: Output format - `"json"`, `"dot"`, `"png"`, `"svg"`, or a list of formats
- `show_memory`: Whether to include memory information
- `output_file`: Base path for file output

**Returns:**
- For single format: the formatted string or graph object
- For multiple formats: the Graph object

### `format_json(graph, show_memory=True)`

Export graph as JSON string.

### `format_dot(graph, show_memory=True)`

Export graph as DOT format string for Graphviz.

### `format_image(graph, output_file, format="png", show_memory=True)`

Export graph as PNG or SVG image (requires `graphviz` package).

### `traverse_graph(output_tensor, input_tensor, graph, visited=None)`

Traverse the computation graph and populate a Graph object.

### `Graph` and `GraphNode`

Data structures for representing the computation graph.

## JSON Output Example

```json
{
  "summary": {
    "num_nodes": 3,
    "total_saved_memory_bytes": 4800,
    "total_saved_memory_formatted": "4,800"
  },
  "nodes": [
    {
      "id": 0,
      "op_type": "AddmmBackward",
      "output_shape": [10, 30],
      "saved_tensors": [...],
      "saved_memory_bytes": 2400,
      "saved_memory_formatted": "2,400"
    }
  ],
  "edges": [...]
}
```

## License

MIT