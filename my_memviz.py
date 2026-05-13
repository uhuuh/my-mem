from dataclasses import dataclass, field
from typing import List, Dict, Any
import warnings
import json
import traceback
import os
from contextlib import contextmanager
import torch


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
    
    factory_functions = [
        'randn', 'rand', 'zeros', 'ones', 'empty', 'full',
        'arange', 'linspace', 'logspace', 'eye', 'tensor',
        'from_numpy', 'as_tensor'
    ]
    
    original_factories = {}
    
    def make_tracked_factory(original_func):
        def tracked_factory(*args, **kwargs):
            tensor = original_func(*args, **kwargs)
            _track_tensor_creation(tensor, captured_tensors)
            return tensor
        return tracked_factory
    
    for name in factory_functions:
        if hasattr(torch, name):
            original_factories[name] = getattr(torch, name)
            setattr(torch, name, make_tracked_factory(original_factories[name]))
    
    def module_forward_hook(module, input, output):
        if isinstance(output, torch.Tensor) and output.grad_fn is not None:
            _track_tensor_creation(output, captured_tensors)
    
    hook_handle = torch.nn.modules.module.register_module_forward_hook(module_forward_hook)
    
    try:
        yield None
    finally:
        for name, original in original_factories.items():
            setattr(torch, name, original)
        hook_handle.remove()
        
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