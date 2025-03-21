import ast
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from collections import defaultdict
from typing import List, Dict, Tuple
import time
import graphviz

# Constants
BITS_PER_MASK = 64
MAX_MASK_COUNT = 10

class OptimizedCFGBuilder(ast.NodeVisitor):
    """
    An optimized Control Flow Graph (CFG) builder that traverses the AST (Abstract Syntax Tree)
    of Python source code and constructs a CFG. It also keeps track of predecessors for each node.
    """
    def __init__(self):
        self.cfg = defaultdict(list)           # adjacency list for CFG
        self.predecessors = defaultdict(list)  # reverse adjacency list for CFG
        self.current_node = 0                  # tracks the current node during traversal
        self.node_count = 0                    # total number of nodes in the CFG
        self.labels = {}                       # human-readable labels for each node
        self.return_nodes = set()              # nodes that are return statements

    def add_edge(self, from_node: int, to_node: int) -> None:
        """
        Add a directed edge from `from_node` to `to_node` in the CFG.
        """
        if to_node not in self.cfg[from_node]:
            self.cfg[from_node].append(to_node)
            self.predecessors[to_node].append(from_node)

    def new_node(self) -> int:
        """
        Create and return a new node ID.
        """
        self.node_count += 1
        return self.node_count - 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Handle the entry point for a function definition in the AST.
        """
        # Reset node tracking for each function definition
        self.current_node = 0
        self.node_count = 1
        self.labels[0] = node

        # Create an entry node that points to the function body
        entry_node = self.new_node()
        self.labels[entry_node] = f"entry_{node.name}"
        self.add_edge(0, entry_node)
        self.current_node = entry_node

        # Visit each statement in the function body
        for stmt in node.body:
            self.visit(stmt)

    def visit_If(self, node: ast.If) -> None:
        """
        Handle 'if' statements by creating 'then' and 'else' branches,
        and linking them to a merge node if necessary.
        """
        condition_node = self.current_node

        # 'then' branch
        then_node = self.new_node()
        self.labels[then_node] = f"then_{node.lineno}"
        self.add_edge(condition_node, then_node)
        self.current_node = then_node

        for stmt in node.body:
            self.visit(stmt)
        then_end = self.current_node

        # 'else' branch
        if node.orelse:
            else_node = self.new_node()
            self.labels[else_node] = f"else_{node.lineno}"
            self.add_edge(condition_node, else_node)
            self.current_node = else_node

            for stmt in node.orelse:
                self.visit(stmt)
            else_end = self.current_node

            # Merge point if neither branch ends in a return
            if then_end not in self.return_nodes and else_end not in self.return_nodes:
                merge_node = self.new_node()
                self.labels[merge_node] = f"merge_{node.lineno}"
                self.add_edge(then_end, merge_node)
                self.add_edge(else_end, merge_node)
                self.current_node = merge_node
        else:
            # Single merge point if 'then' branch doesn't return
            if then_end not in self.return_nodes:
                merge_node = self.new_node()
                self.labels[merge_node] = f"merge_{node.lineno}"
                self.add_edge(then_end, merge_node)
                self.add_edge(condition_node, merge_node)
                self.current_node = merge_node

    def visit_Return(self, node: ast.Return) -> None:
        """
        Handle 'return' statements by creating a dedicated node and marking it as a return node.
        """
        return_node = self.new_node()
        self.labels[return_node] = node
        self.add_edge(self.current_node, return_node)
        self.current_node = return_node
        self.return_nodes.add(return_node)

    def _process_loop(self, node: ast.AST, loop_type: str) -> None:
        """
        Helper method to handle loops (for and while). Creates a header, body, and exit node,
        plus an else-block if needed.
        """
        # Header node
        header_node = self.new_node()
        self.labels[header_node] = f"{loop_type}_header_{node.lineno}"
        self.add_edge(self.current_node, header_node)
        self.current_node = header_node

        # Body node
        body_node = self.new_node()
        self.labels[body_node] = f"{loop_type}_body_{node.lineno}"
        self.add_edge(header_node, body_node)
        self.current_node = body_node

        for stmt in node.body:
            self.visit(stmt)

        # Go back to header from the end of the body
        self.add_edge(self.current_node, header_node)

        # Exit node
        exit_node = self.new_node()
        self.labels[exit_node] = f"{loop_type}_exit_{node.lineno}"
        self.add_edge(header_node, exit_node)
        self.current_node = exit_node

        # 'else' block (only for Python 'for'/'while' else clauses)
        if node.orelse:
            else_node = self.new_node()
            self.labels[else_node] = f"{loop_type}_else_{node.lineno}"
            self.add_edge(exit_node, else_node)
            self.current_node = else_node

            for stmt in node.orelse:
                self.visit(stmt)

    def visit_For(self, node: ast.For) -> None:
        """
        Handle 'for' loops using the helper loop-processing method.
        """
        self._process_loop(node, "for")

    def visit_While(self, node: ast.While) -> None:
        """
        Handle 'while' loops using the helper loop-processing method.
        """
        self._process_loop(node, "while")

    def generic_visit(self, node: ast.AST) -> None:
        """
        Default AST visitor that creates a new node for each AST node not handled by specific methods.
        """
        if isinstance(node, (ast.FunctionDef, ast.If, ast.For, ast.While, ast.Return)):
            # Already handled in specific visit methods
            pass
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, ast.AST):
                    self.visit(item)
        else:
            next_node = self.new_node()
            self.labels[next_node] = type(node).__name__
            self.add_edge(self.current_node, next_node)
            self.current_node = next_node

            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            self.visit(item)
                elif isinstance(value, ast.AST):
                    self.visit(value)

def compute_dominators_optimized(source_code: str) -> Tuple[np.ndarray, Dict[int, List[int]], float]:
    """
    Build the CFG from the given Python source code, then compute dominators using a GPU-accelerated approach.
    
    :param source_code: The Python source code string to be analyzed.
    :return: A tuple of:
        - dom (np.ndarray): The dominator bitmasks for each node.
        - tree (Dict[int, List[int]]): The dominator tree as a dictionary {idom_node: [dominated_nodes]}.
        - elapsed_time (float): Time taken to compute dominators (in milliseconds).
    """
    # 1. Build the CFG
    parsed_ast = ast.parse(source_code)
    builder = OptimizedCFGBuilder()
    builder.visit(parsed_ast)
    num_nodes = builder.node_count

    # 2. Prepare predecessors for GPU processing
    predecessor_list = []
    predecessor_offsets = [0]

    for node_id in range(num_nodes):
        predecessor_list.extend(builder.predecessors[node_id])
        predecessor_offsets.append(len(predecessor_list))

    predecessors = np.array(predecessor_list, dtype=np.int32)
    predecessor_offsets = np.array(predecessor_offsets, dtype=np.int32)
    num_preds = len(predecessors)

    # 3. Calculate how many 64-bit masks are needed to represent dominators
    mask_count = (num_nodes + BITS_PER_MASK - 1) // BITS_PER_MASK
    if mask_count > MAX_MASK_COUNT:
        raise ValueError(f"mask_count ({mask_count}) exceeds MAX_MASK_COUNT ({MAX_MASK_COUNT}).")

    # 4. Initialize dominator sets
    #    Each node's dominator set is represented by 'mask_count' 64-bit integers.
    dom_matrix = np.zeros((num_nodes, mask_count), dtype=np.uint64)
    # Entry node (node 0) dominates itself
    dom_matrix[0, 0] = 1

    # 5. Copy data to GPU
    start_time = time.time()
    predecessors_gpu = cuda.to_device(predecessors)
    predecessor_offsets_gpu = cuda.to_device(predecessor_offsets)
    dom_gpu = cuda.to_device(dom_matrix)

    changed_flag = cuda.mem_alloc(np.zeros(1, dtype=np.int32).nbytes)

    # 6. CUDA kernel
    module = SourceModule(
        f"""
        #define BITS_PER_MASK {BITS_PER_MASK}
        #define MAX_MASK_COUNT {MAX_MASK_COUNT}

        __global__ void compute_dominator_optimized(
            unsigned long long *dom,
            int *predecessors,
            int *pred_offsets,
            int num_nodes,
            int num_preds,
            int mask_count,
            int *d_changed)
        {{
            int node = blockIdx.x * blockDim.x + threadIdx.x;
            if (node >= num_nodes) return;
            if (mask_count > MAX_MASK_COUNT) return;

            // The entry node (node 0) is handled separately
            if (node == 0) {{
                // Root node: it dominates only itself
                dom[0] = 1ULL;
                for (int i = 1; i < mask_count; i++) {{
                    dom[i] = 0ULL;
                }}
                return;
            }}

            // Each thread might be in a warp; we store intersection results in shared mem
            __shared__ unsigned long long shared_intersection[MAX_MASK_COUNT * 32];
            int warp_id = threadIdx.x / 32;
            int lane_id = threadIdx.x % 32;
            unsigned long long *my_intersection = &shared_intersection[warp_id * MAX_MASK_COUNT];

            // Initialize intersection to all 1s (bitwise)
            #pragma unroll
            for (int i = 0; i < mask_count; i++) {{
                if (lane_id == 0) {{
                    my_intersection[i] = ~0ULL;
                }}
            }}
            __syncwarp();

            // Compute intersection of all predecessors
            int start = pred_offsets[node];
            int end = (node + 1 < num_nodes) ? pred_offsets[node + 1] : num_preds;

            for (int i = start; i < end; i++) {{
                int pred = predecessors[i];
                #pragma unroll
                for (int j = 0; j < mask_count; j++) {{
                    my_intersection[j] &= dom[pred * mask_count + j];
                }}
            }}

            // Include this node itself in the dominator set
            int mask_index = node / BITS_PER_MASK;
            int bit_index = node % BITS_PER_MASK;
            if (mask_index < mask_count) {{
                my_intersection[mask_index] |= (1ULL << bit_index);
            }}

            // Check if the new dominator set differs from the old one
            bool changed = false;
            #pragma unroll
            for (int i = 0; i < mask_count; i++) {{
                if (dom[node * mask_count + i] != my_intersection[i]) {{
                    changed = true;
                    break;
                }}
            }}

            // If changed, update and notify
            if (changed) {{
                #pragma unroll
                for (int i = 0; i < mask_count; i++) {{
                    dom[node * mask_count + i] = my_intersection[i];
                }}
                atomicExch(d_changed, 1);
            }}
        }}
        """
    )

    compute_dominator_kernel = module.get_function("compute_dominator_optimized")

    block_size = 256
    grid_size = (num_nodes + block_size - 1) // block_size

    # 7. Iteratively update until convergence or maximum iterations reached
    iterations = 0
    max_iterations = 100

    while iterations < max_iterations:
        iterations += 1
        host_flag = np.zeros(1, dtype=np.int32)
        cuda.memcpy_htod(changed_flag, host_flag)

        compute_dominator_kernel(
            dom_gpu,
            predecessors_gpu,
            predecessor_offsets_gpu,
            np.int32(num_nodes),
            np.int32(num_preds),
            np.int32(mask_count),
            changed_flag,
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )

        cuda.memcpy_dtoh(host_flag, changed_flag)
        if host_flag[0] == 0:
            # No changes, so we've converged
            break

    # 8. Copy final dominator results back to host
    cuda.memcpy_dtoh(dom_matrix, dom_gpu)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000.0  # milliseconds

    # 9. Build the dominator tree (a simplistic approach)
    dom_tree = defaultdict(list)

    # For each node (excluding the root), find the immediate dominator
    for node in range(1, num_nodes):
        idom = None
        for candidate in range(num_nodes):
            if candidate == node:
                continue

            # Check if 'candidate' actually dominates 'node'
            mask_idx = candidate // BITS_PER_MASK
            bit_idx = candidate % BITS_PER_MASK
            if mask_idx >= mask_count:
                continue

            # For readability
            node_mask = np.uint64(dom_matrix[node][mask_idx])
            candidate_bit = np.uint64(1 << bit_idx)

            if (node_mask & candidate_bit) != 0:
                # If 'candidate' is a dominator, pick the closest one (small heuristic)
                if idom is None:
                    idom = candidate
                else:
                    # If 'candidate' is dominated by the current 'idom', then 'candidate' is closer
                    # This logic can be refined for more precise "immediate dominator" definitions
                    # but is kept simple for illustrative purposes.
                    is_closer = True
                    for j in range(mask_count):
                        # We check if all bits set in dom_matrix[candidate] are also set in dom_matrix[idom].
                        # If so, candidate is "below" idom in the dominator hierarchy.
                        if (dom_matrix[idom][j] & dom_matrix[candidate][j]) != dom_matrix[candidate][j]:
                            is_closer = False
                            break
                    if is_closer:
                        idom = candidate

        # Once the immediate dominator is identified, add to dominator tree
        if idom is not None:
            dom_tree[idom].append(node)

    return dom_matrix, dom_tree, elapsed_time

def visualize_cfg(cfg: Dict[int, List[int]], labels: Dict[int, str], filename='cfg', file_format='png'):
    """
    Visualize the Control Flow Graph using graphviz.

    :param cfg: A dictionary representing the CFG adjacency list {node_id: [successor_ids]}.
    :param labels: A dictionary of labels for each node {node_id: label_string}.
    :param filename: Output filename (without extension) for the generated graph.
    :param file_format: The file format for the rendered graph (e.g. 'png', 'pdf', etc.).
    :return: The generated Graphviz object.
    """
    dot = graphviz.Digraph(comment='Control Flow Graph', format=file_format)
    dot.attr(rankdir='TB', size='8,8', dpi='300')

    # Define different styles for various node types
    node_styles = {
        'function': {'shape': 'oval', 'fillcolor': '#FFDDDD'},
        'conditional': {'shape': 'diamond', 'fillcolor': '#DDFFDD'},
        'loop': {'shape': 'hexagon', 'fillcolor': '#DDDDFF'},
        'return': {'shape': 'box', 'fillcolor': '#FFDDFF'},
        'default': {'shape': 'box', 'fillcolor': 'white'}
    }

    # Add nodes with appropriate style
    for node_id in sorted(labels.keys()):
        label_str = str(labels[node_id])

        # Determine node type from the label
        node_type = 'default'
        if 'FunctionDef' in label_str:
            node_type = 'function'
        elif any(x in label_str for x in ['if', 'then', 'else']):
            node_type = 'conditional'
        elif any(x in label_str for x in ['for', 'while']):
            node_type = 'loop'
        elif 'Return' in label_str:
            node_type = 'return'

        style = node_styles[node_type]
        dot.node(
            str(node_id),
            label=f"{node_id}: {label_str}",
            shape=style['shape'],
            style='filled',
            fillcolor=style['fillcolor']
        )

    # Add edges
    for from_node, to_nodes in cfg.items():
        for to_node in to_nodes:
            dot.edge(str(from_node), str(to_node))

    # Render the graph
    dot.render(filename, view=False)
    return dot

if __name__ == "__main__":
    # Example Python code to analyze
    example_source_code = """
def complex_example(a, b, c, d, e):
    x = 0
    y = 0

    if a > b:
        if b > c:
            x = a + b
            while d > 0:
                x += 1
                d -= 1
        else:
            x = a - b
            for i in range(10):
                if i % 2 == 0:
                    x += i
    else:
        if a > c:
            y = b + c
            for j in range(e):
                if j > 5:
                    break
                y += j
        else:
            if c > 10:
                y = 100
                while e > 0:
                    y -= 1
                    e -= 1
            else:
                y = -100

    if x > y:
        return x
    else:
        return y
"""

    # 1. Build and visualize the CFG
    cfg_builder = OptimizedCFGBuilder()
    ast_tree = ast.parse(example_source_code)
    cfg_builder.visit(ast_tree)
    visualize_cfg(cfg_builder.cfg, cfg_builder.labels, filename='optimized_cfg')

    # 2. Compute dominators
    dominators, dominator_tree, time_ms = compute_dominators_optimized(example_source_code)
    print("\nDominator Tree:")
    for parent, children in dominator_tree.items():
        print(f"Node {parent} → {sorted(children)}")

    print(f"\nTime to compute dominators: {time_ms:.3f} ms")
