from dataclasses import dataclass, field
from typing import List, Dict, Any
import warnings
import json
import traceback
import os
from contextlib import contextmanager


def format_bytes(num_bytes):
    if num_bytes == 0:
        return "0"
    return "{:,}".format(num_bytes)


def calculate_tensor_memory(tensor):
    return tensor.nelement() * tensor.element_size()


def extract_saved_tensors(grad_fn):
    saved_tensors = []
    
    for attr_name in dir(grad_fn):
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


@dataclass
class GraphNode:
    node_id: int
    op_type: str
    output_shape: List[int]
    saved_tensors: List[Dict[str, Any]] = field(default_factory=list)
    saved_memory_bytes: int = 0
    call_stack: List[Dict[str, Any]] = field(default_factory=list)


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

    def render(self, format: str, show_memory: bool = True, output_file: str = None):
        if format == "json":
            return format_json(self, show_memory)
        elif format == "dot":
            return format_dot(self, show_memory)
        elif format in ["png", "svg"]:
            if output_file is None:
                warnings.warn(f"output_file required for {format} format")
                return None
            return format_image(self, output_file, format=format, show_memory=show_memory)
        return None


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
            return None
        
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
        
        for next_fn, _ in grad_fn.next_functions:
            if next_fn is not None and not _is_target_node(next_fn):
                child_id = _traverse(next_fn, visited)
                if child_id is not None:
                    graph.add_edge(node_id, child_id)
        
        return node_id
    
    _traverse(output_tensor.grad_fn, visited)
    
    if len(graph.nodes) == 0:
        warnings.warn("No path found between tensors")


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
    
    for fmt in formats:
        if fmt in ["png", "svg"]:
            if output_file is None:
                warnings.warn(f"output_file required for {fmt} format")
                continue
            
            ext = f".{fmt}"
            if output_file.endswith(f".{fmt}"):
                out_path = output_file
            else:
                out_path = output_file + ext
            
            graph.render(fmt, show_memory=show_memory, output_file=out_path)
        elif output_file is not None:
            result = graph.render(fmt, show_memory=show_memory)
            if result is not None:
                ext = f".{fmt}"
                if not output_file.endswith(ext):
                    path = output_file + ext
                else:
                    path = output_file
                with open(path, 'w') as f:
                    f.write(result)
    
    return graph