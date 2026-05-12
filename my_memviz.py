from dataclasses import dataclass, field
from typing import List, Dict, Any
import warnings


def format_bytes(num_bytes):
    if num_bytes == 0:
        return "0"
    return "{:,}".format(num_bytes)


def calculate_tensor_memory(tensor):
    return tensor.nelement() * tensor.element_size()


def extract_saved_tensors(grad_fn):
    saved_tensors = []
    
    for attr_name in dir(grad_fn):
        if attr_name.startswith('_saved_') and not attr_name.startswith('_saved_') and attr_name != '_saved':
            continue
        if attr_name.startswith('_saved_') and not attr_name.startswith('_raw_saved_'):
            if attr_name.startswith('_saved_mat') or attr_name == '_saved_result' or attr_name == '_saved_bias' or attr_name == '_saved_weight':
                try:
                    tensor = getattr(grad_fn, attr_name)
                    if tensor is not None and hasattr(tensor, 'shape'):
                        name = attr_name.replace('_saved_', '')
                        saved_tensors.append({
                            "name": name,
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype).replace("torch.", ""),
                            "size_bytes": calculate_tensor_memory(tensor),
                            "size_formatted": format_bytes(calculate_tensor_memory(tensor))
                        })
                except RuntimeError:
                    pass
    
    return saved_tensors


@dataclass
class GraphNode:
    node_id: int
    op_type: str
    output_shape: List[int]
    saved_tensors: List[Dict[str, Any]] = field(default_factory=list)
    saved_memory_bytes: int = 0


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