from dataclasses import dataclass, field
from typing import List, Dict, Any


def format_bytes(num_bytes):
    if num_bytes == 0:
        return "0"
    return "{:,}".format(num_bytes)


def calculate_tensor_memory(tensor):
    return tensor.nelement() * tensor.element_size()


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