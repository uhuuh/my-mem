from dataclasses import dataclass, field
from typing import List, Dict, Any


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