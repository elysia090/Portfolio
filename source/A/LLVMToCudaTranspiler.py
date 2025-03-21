import unittest
import llvmlite.binding as llvm
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize LLVM
llvm.initialize()
llvm.initialize_all_targets()
llvm.initialize_all_asmprinters()

# Type definitions
LLVMType = Any  # Type for LLVM IR types
CUDAType = str  # Type for CUDA C types
BlockName = str  # Type for basic block names
DominatorSet = Set[BlockName]  # Type for dominator sets
PTXCode = str  # Type for PTX assembly code
KernelFunction = Any  # Type for CUDA kernel functions

# Constants
BITS_PER_MASK: int = 64  # Number of bits per mask for dominator computation
MAX_MASK_COUNT: int = 10  # Maximum number of masks allowed


class LLVMToCudaTranspiler:
    """LLVM IR to CUDA transpiler with end-to-end execution capabilities.
    
    This class handles conversion of LLVM IR code to CUDA code and provides
    functionality for executing the converted code on GPU.
    
    The transpiler supports:
    - Converting LLVM IR to CUDA C code
    - Converting LLVM IR to PTX assembly
    - Computing dominators using CUDA parallelization
    - End-to-end execution of LLVM IR on GPU
    - CPU-based validation execution
    
    Attributes:
        target_triple (str): Target architecture triple (e.g. "nvptx64-nvidia-cuda")
        data_layout (str): Data layout string for LLVM
        gpu_arch (str): GPU architecture version (e.g. "sm_70") 
        opt_level (int): Optimization level (0-3)
        kernel_cache (Dict[str, Any]): Cache of compiled CUDA kernels
        
    Example:
        transpiler = LLVMToCudaTranspiler()
        cuda_code = transpiler.transpile_to_cuda(llvm_ir)
        result = transpiler.end_to_end_execute(llvm_ir, input_data)
    """

    # Type mapping from LLVM IR types to CUDA C types
    TYPE_MAP: Dict[str, str] = {
        "i1": "bool", "i8": "char", "i16": "short", "i32": "int", "i64": "long long",
        "float": "float", "double": "double", "void": "void"
    }
    
    # Operation mapping from LLVM IR operations to CUDA C operators
    OP_MAPS: Dict[str, str] = {
        # Binary operations
        "add": "+", "fadd": "+", "sub": "-", "fsub": "-", 
        "mul": "*", "fmul": "*", "sdiv": "/", "udiv": "/", "fdiv": "/",
        # Comparison predicates
        "eq": "==", "ne": "!=", "sgt": ">", "sge": ">=", "slt": "<", "sle": "<=", 
        "ugt": ">", "uge": ">=", "ult": "<", "ule": "<="
    }

    def __init__(self, target_triple: str = "nvptx64-nvidia-cuda", 
                 data_layout: str = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",
                 gpu_arch: str = "sm_70", opt_level: int = 2) -> None:
        """Initialize the transpiler with target configuration.
        
        Args:
            target_triple: Target architecture triple for LLVM
            data_layout: Data layout string for LLVM
            gpu_arch: GPU architecture version (e.g. sm_70)
            opt_level: Optimization level (0-3)
            
        The target triple and data layout specify how LLVM should generate code.
        The GPU architecture version determines which GPU features can be used.
        The optimization level controls how aggressively LLVM optimizes the code.
        """
        self.target_triple: str = target_triple
        self.data_layout: str = data_layout
        self.gpu_arch: str = gpu_arch
        self.opt_level: int = opt_level
        self.kernel_cache: Dict[str, Any] = {}
        logger.info(f"Initialized transpiler with target: {target_triple}, GPU arch: {gpu_arch}")

    @lru_cache(maxsize=32)
    def parse_module(self, llvm_ir: str) -> Any:
        """Parse LLVM IR into a module (with caching).
        
        This method parses LLVM IR text into an LLVM module object. Results are cached
        to avoid reparsing the same IR multiple times.
        
        Args:
            llvm_ir: LLVM IR code string
            
        Returns:
            Parsed LLVM module
            
        Raises:
            llvm.LLVMException: If parsing fails or module verification fails
        """
        mod = llvm.parse_assembly(llvm_ir)
        mod.verify()
        
        # Set target triple and data layout if not already set
        if not mod.triple:
            mod.triple = self.target_triple
        if not mod.data_layout:
            mod.data_layout = self.data_layout
        return mod

    def compute_dominators(self, llvm_ir: str) -> Tuple[Dict[str, Set[str]], List[str]]:
        """Compute dominators from LLVM IR using CUDA parallelization.
        
        This method computes the dominator sets for each basic block in the LLVM IR
        using a parallel algorithm implemented in CUDA. A dominator set for a block B
        contains all blocks that dominate B (must be executed before B).
        
        Args:
            llvm_ir: LLVM IR code string
            
        Returns:
            Tuple containing:
                - Dictionary mapping block names to their dominator sets
                - List of block names in order
                
        Raises:
            ValueError: If no non-declaration function found or mask count exceeds limit
            
        The algorithm uses a bit vector representation for efficient computation on GPU.
        Each block's dominator set is represented as a bit vector where bit i is set if
        block i dominates the block.
        """
        # Parse LLVM IR and extract function
        mod = self.parse_module(llvm_ir)
        func = next((f for f in mod.functions if not f.is_declaration), None)
        if not func:
            raise ValueError("No non-declaration function found in IR")

        # Build CFG and extract blocks
        cfg_succ, block_names = self._extract_cfg(func)
        num_nodes: int = len(block_names)

        # Process predecessors for dominator computation
        pred_index_list, pred_offsets = self._build_predecessors(cfg_succ, block_names)

        # Set up bit vector representation
        mask_count: int = (num_nodes + BITS_PER_MASK - 1) // BITS_PER_MASK
        if mask_count > MAX_MASK_COUNT:
            raise ValueError(f"mask_count ({mask_count}) exceeds MAX_MASK_COUNT ({MAX_MASK_COUNT})")

        # Prepare data for CUDA processing
        dom_matrix, _ = self._run_dominator_kernel(num_nodes, pred_index_list, pred_offsets, mask_count)
        
        # Convert bit vector representation to dominator sets
        dom_sets: Dict[str, Set[str]] = {}
        for i, node_name in enumerate(block_names):
            dom_set: Set[str] = set()
            for j in range(num_nodes):
                mask_idx: int = j // BITS_PER_MASK
                bit_idx: int = j % BITS_PER_MASK
                if (dom_matrix[i, mask_idx] & (1 << bit_idx)) != 0:
                    dom_set.add(block_names[j])
            dom_sets[node_name] = dom_set
            
        return dom_sets, block_names

    def _extract_cfg(self, func: Any) -> Tuple[Dict[str, List[str]], List[str]]:
        """Extract control flow graph from function.
        
        This method builds a control flow graph (CFG) from an LLVM function by
        analyzing the terminator instructions of each basic block.
        
        Args:
            func: LLVM function
            
        Returns:
            Tuple containing:
                - Dictionary mapping block names to successor lists
                - List of block names in order
                
        The CFG is represented as an adjacency list where each block maps to
        its list of successor blocks. Supported terminator instructions include:
        - Unconditional branches (br)
        - Conditional branches (br with condition)
        - Switch statements
        """
        cfg_succ: Dict[str, List[str]] = {}
        for block in func.blocks:
            name: str = block.name
            succ_list: List[str] = []
            instructions = list(block.instructions)
            if not instructions:
                continue

            term = instructions[-1]  # terminator instruction
            if term.opcode == "br":
                ops = list(term.operands)
                if len(ops) == 3:  # conditional branch
                    succ_list.extend(op.name for op in ops[1:])
                elif len(ops) == 1:  # unconditional branch
                    succ_list.append(ops[0].name)
            elif term.opcode == "switch":
                ops = list(term.operands)
                if len(ops) >= 2:
                    succ_list.append(ops[1].name)  # default target
                    # add case targets
                    for i in range(2, len(ops), 2):
                        if i+1 < len(ops):
                            succ_list.append(ops[i+1].name)

            cfg_succ[name] = succ_list

        block_names: List[str] = list(cfg_succ.keys())
        return cfg_succ, block_names

    def _build_predecessors(self, cfg_succ: Dict[str, List[str]], block_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Build predecessor lists for dominator computation.
        
        This method converts the CFG successor lists into predecessor lists and
        flattens them into arrays suitable for GPU processing.
        
        Args:
            cfg_succ: Dictionary mapping block names to successor lists
            block_names: List of block names in order
            
        Returns:
            Tuple containing:
                - Array of predecessor indices (flattened)
                - Array of predecessor offsets into the flattened array
                
        The predecessor information is stored in two arrays:
        - A flattened array containing all predecessor indices
        - An offset array indicating where each block's predecessors start
        This format allows efficient access on GPU.
        """
        index_of: Dict[str, int] = {name: i for i, name in enumerate(block_names)}
        num_nodes: int = len(block_names)
        preds: List[List[int]] = [[] for _ in range(num_nodes)]

        for src, succ_list in cfg_succ.items():
            for tgt in succ_list:
                preds[index_of[tgt]].append(index_of[src])

        # Convert preds lists to flattened array + offsets
        pred_index_list: List[int] = []
        pred_offsets: List[int] = []
        for i in range(num_nodes):
            pred_offsets.append(len(pred_index_list))
            pred_index_list.extend(preds[i])

        return np.array(pred_index_list, dtype=np.int32), np.array(pred_offsets, dtype=np.int32)

    def _run_dominator_kernel(self, num_nodes: int, predecessors: np.ndarray, pred_offsets: np.ndarray, 
                            mask_count: int) -> Tuple[np.ndarray, List[str]]:
        """Run CUDA kernel for dominator computation.
        
        This method executes the dominator computation kernel on the GPU. The kernel
        iteratively computes dominator sets until convergence using a parallel algorithm.
        
        Args:
            num_nodes: Number of nodes in the graph
            predecessors: Array of predecessor indices
            pred_offsets: Array of predecessor offsets
            mask_count: Number of bit masks needed
            
        Returns:
            Tuple containing:
                - Dominator matrix (bit vectors)
                - List of block names
                
        The kernel uses a bit vector representation where each block's dominators
        are stored as bits in an array of 64-bit integers. The kernel iteratively
        updates these bit vectors until no changes occur.
        
        The algorithm is based on the iterative dominator computation:
        dom(n) = {n} ∪ (∩ dom(p) for p in preds(n))
        """
        # Initialize dominator matrix
        dom: np.ndarray = np.zeros((num_nodes, mask_count), dtype=np.uint64)
        dom[0, 0] = 1 << 0  # entry node dominates itself

        # Prepare device memory
        pinned_offs = cuda.pagelocked_empty_like(pred_offsets)
        pinned_offs[:] = pred_offsets
        d_offs = cuda.to_device(pinned_offs)

        pinned_dom = cuda.pagelocked_empty_like(dom)
        pinned_dom[:] = dom
        d_dom = cuda.to_device(pinned_dom)

        d_changed = cuda.mem_alloc(np.int32(0).nbytes)

        # Handle the case when there are no predecessors
        num_preds: int = len(predecessors)
        if num_preds > 0:
            pinned_pred = cuda.pagelocked_empty_like(predecessors)
            pinned_pred[:] = predecessors
            d_pred = cuda.to_device(pinned_pred)
        else:
            d_pred = cuda.to_device(np.array([0], dtype=np.int32))
            num_preds = 1

        # Get or create kernel
        if "dominator_kernel" not in self.kernel_cache:
            module = self._create_dominator_kernel_module()
            self.kernel_cache["dominator_kernel"] = module.get_function("compute_dominator_optimized")

        kernel = self.kernel_cache["dominator_kernel"]

        # Launch configuration
        block_size: int = 256
        grid_size: int = (num_nodes + block_size - 1) // block_size

        # Iterate until convergence
        for iteration in range(1000):
            host_changed = np.zeros(1, dtype=np.int32)
            cuda.memcpy_htod(d_changed, host_changed)

            kernel(d_dom, d_pred, d_offs,
                  np.int32(num_nodes), np.int32(num_preds), np.int32(mask_count),
                  d_changed,
                  block=(block_size,1,1), grid=(grid_size,1,1))

            cuda.memcpy_dtoh(host_changed, d_changed)
            if host_changed[0] == 0:
                logger.info(f"Dominator computation converged after {iteration+1} iterations")
                break

        # Get results
        cuda.memcpy_dtoh(pinned_dom, d_dom)
        return np.array(pinned_dom, copy=True), block_names

    def _create_dominator_kernel_module(self) -> SourceModule:
        """Create CUDA kernel module for dominator computation.
        
        This method generates the CUDA C code for the dominator computation kernel
        and compiles it into a module.
        
        Returns:
            CUDA source module containing the dominator kernel
            
        The kernel implements a parallel algorithm where each thread handles one node
        in the graph. For each node, it:
        1. Intersects the dominator sets of all predecessors
        2. Adds the node itself to its dominator set
        3. Updates the result if changed
        
        The kernel uses shared memory for efficient warp-level operations and
        atomic operations for detecting convergence.
        """
        cuda_code = f"""
        #define BITS_PER_MASK {BITS_PER_MASK}
        #define MAX_MASK_COUNT {MAX_MASK_COUNT}

        __global__ void compute_dominator_optimized(unsigned long long *dom,
                                                const int *preds, const int *pred_offs,
                                                int num_nodes, int num_preds, int mask_count,
                                                int *d_changed) {{
            int node = blockIdx.x * blockDim.x + threadIdx.x;
            if (node >= num_nodes) return;
            if (mask_count > MAX_MASK_COUNT) return;

            // Entry node special case
            if (node == 0) {{
                for(int j=0; j<mask_count; ++j) dom[j] = 0ULL;
                dom[0] = 1ULL << 0;
                return;
            }}

            // Shared memory for warp-level reduction
            __shared__ unsigned long long shared_intersection[MAX_MASK_COUNT * 32];
            int warp_id = threadIdx.x / 32;
            int lane_id = threadIdx.x % 32;
            unsigned long long *my_intersection = &shared_intersection[warp_id * mask_count];

            // Initialize intersection to all 1s
            for(int j=0; j<mask_count; ++j) {{
                if(lane_id == 0) my_intersection[j] = 0xFFFFFFFFFFFFFFFFULL;
            }}
            __syncwarp();

            // Intersect all predecessors' dominator sets
            int start = pred_offs[node];
            int end = (node + 1 < num_nodes) ? pred_offs[node + 1] : num_preds;

            for(int i = start; i < end; ++i) {{
                int pred = preds[i];
                for(int j=0; j<mask_count; ++j) {{
                    my_intersection[j] &= dom[pred * mask_count + j];
                }}
            }}

            // A node always dominates itself
            int mask_index = node / BITS_PER_MASK;
            int bit_index = node % BITS_PER_MASK;
            if(mask_index < mask_count) {{
                my_intersection[mask_index] |= (1ULL << bit_index);
            }}

            // Update dominator set if changed
            bool changed = false;
            for(int j=0; j<mask_count; ++j) {{
                unsigned long long newval = my_intersection[j];
                if(dom[node * mask_count + j] != newval) {{
                    dom[node * mask_count + j] = newval;
                    changed = true;
                }}
            }}

            if(changed) atomicExch(d_changed, 1);
        }}
        """
        return SourceModule(cuda_code, no_extern_c=True)

    def llvm_to_ptx(self, llvm_ir: str) -> str:
        """Convert LLVM IR to PTX assembly for CUDA execution.
        
        This method converts LLVM IR to PTX assembly code that can be executed
        on NVIDIA GPUs.
        
        Args:
            llvm_ir: LLVM IR code string
            
        Returns:
            PTX assembly code string
            
        The conversion process:
        1. Parses the LLVM IR
        2. Creates a target machine for the GPU architecture
        3. Generates PTX assembly optimized for the target
        """
        mod = self.parse_module(llvm_ir)
        target = llvm.Target.from_triple(self.target_triple)
        target_machine = target.create_target_machine(
            triple=self.target_triple, cpu=self.gpu_arch, features="",
            opt=self.opt_level, reloc="pic", code_model="default"
        )
        return target_machine.emit_assembly(mod)

    def extract_kernels(self, llvm_ir: str) -> Dict[str, str]:
        """Extract kernel functions from LLVM IR.
        
        This method identifies and extracts CUDA kernel functions from LLVM IR
        by looking for kernel attributes.
        
        Args:
            llvm_ir: LLVM IR code string
            
        Returns:
            Dictionary mapping kernel names to their LLVM IR code
            
        A function is considered a kernel if it has attributes containing
        "kernel" or "global". These attributes are added by the CUDA compiler
        to mark functions that can be called from the host.
        """
        mod = self.parse_module(llvm_ir)
        kernels: Dict[str, str] = {}
        for func in mod.functions:
            if func.is_declaration:
                continue
            is_kernel = any("kernel" in str(attr).lower() or "global" in str(attr).lower()
                          for attr in func.attributes)
            if is_kernel:
                kernels[func.name] = str(func)
        return kernels

    def transpile_to_cuda(self, llvm_ir: str) -> str:
        """Transpile LLVM IR to CUDA C code.
        
        This method converts LLVM IR to equivalent CUDA C code by translating
        each LLVM instruction to CUDA C statements.
        
        Args:
            llvm_ir: LLVM IR code string
            
        Returns:
            CUDA C code string
            
        The transpilation process:
        1. Parses the LLVM IR module
        2. Processes each function
        3. Converts function arguments and return types
        4. Translates each instruction to CUDA C
        5. Handles kernel attributes
        """
        mod = self.parse_module(llvm_ir)
        cuda_code: List[str] = ["// Auto-generated CUDA code from LLVM IR",
                    "#include <cuda_runtime.h>", ""]

        # Process functions
        for func in mod.functions:
            if func.is_declaration:
                continue

            is_kernel = any("kernel" in str(attr).lower() or "global" in str(attr).lower()
                           for attr in func.attributes)

            return_type = self._convert_llvm_type(func.return_type)
            args = [f"{self._convert_llvm_type(arg.type)} {arg.name}" for arg in func.arguments]

            prefix = "__global__" if is_kernel else ""
            cuda_code.append(f"{prefix} {return_type} {func.name}({', '.join(args)}) {{")
            self._process_function_body(func, cuda_code)
            cuda_code.append("}")
            cuda_code.append("")

        return "\n".join(cuda_code)

    def _process_function_body(self, func: Any, cuda_code: List[str]) -> None:
        """Process function body and convert to CUDA.
        
        This method converts the body of an LLVM function to CUDA C code
        by processing each basic block and instruction.
        
        Args:
            func: LLVM function
            cuda_code: List to append CUDA code lines to
            
        The conversion process:
        1. Generates labels for basic blocks
        2. Processes each instruction in each block
        3. Maintains proper indentation and formatting
        """
        block_labels: Dict[str, str] = {block.name: f"block_{i}" for i, block in enumerate(func.blocks)}
        for i, block in enumerate(func.blocks):
            if i > 0:
                cuda_code.append(f"  {block_labels[block.name]}:")
            for instr in block.instructions:
                cuda_instr = self._convert_instruction(instr, block_labels)
                if cuda_instr:
                    cuda_code.append(f"  {cuda_instr}")

    @lru_cache(maxsize=64)
    def _convert_llvm_type(self, llvm_type: LLVMType) -> CUDAType:
        """Convert LLVM types to CUDA C types with caching.
        
        This method converts LLVM IR types to equivalent CUDA C types,
        caching results for efficiency.
        
        Args:
            llvm_type: LLVM type
            
        Returns:
            CUDA C type string
            
        Handles:
        - Basic types (int, float, etc.)
        - Pointer types
        - Vector types
        - Falls back to void* for unknown types
        """
        type_str = str(llvm_type)

        # Check basic type mapping
        for llvm_t, cuda_t in self.TYPE_MAP.items():
            if llvm_t in type_str:
                return cuda_t

        # Handle pointers
        if "*" in type_str:
            base_type = self._convert_llvm_type(llvm_type.pointee)
            return f"{base_type}*"

        # Vector types
        if "x" in type_str and "[" in type_str:
            try:
                vec_match = type_str.split("x")[0]
                count = int(vec_match)
                elem_type = self._convert_llvm_type(llvm_type.element)
                return f"{elem_type}{count}"
            except (ValueError, AttributeError):
                pass

        return "void*"  # Default fallback

    def _convert_instruction(self, instr: Any, block_labels: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Convert a single LLVM instruction to CUDA C code.
        
        This method converts an LLVM instruction to equivalent CUDA C code,
        handling various instruction types.
        
        Args:
            instr: LLVM instruction
            block_labels: Dictionary mapping block names to labels
            
        Returns:
            CUDA C code string or None if instruction cannot be converted
            
        Handles:
        - Return instructions
        - Binary operations
        - Comparisons
        - Control flow (branches)
        - Memory operations
        - Function calls
        - Type conversions
        - PHI nodes (simplified)
        """
        if block_labels is None:
            block_labels = {}
        opcode = instr.opcode

        # Return instruction
        if opcode == "ret":
            return f"return {instr.operands[0].name};" if len(instr.operands) > 0 else "return;"

        # Binary operations
        if opcode in self.OP_MAPS and len(instr.operands) >= 2:
            return f"{instr.name} = {instr.operands[0].name} {self.OP_MAPS[opcode]} {instr.operands[1].name};"

        # Comparisons
        elif opcode in ["icmp", "fcmp"] and len(instr.operands) >= 2:
            instr_str = str(instr)
            for pred, op in self.OP_MAPS.items():
                if pred in instr_str:
                    return f"{instr.name} = {instr.operands[0].name} {op} {instr.operands[1].name};"

        # Control flow
        elif opcode == "br":
            if len(instr.operands) == 1:  # Unconditional
                label = block_labels.get(instr.operands[0].name, instr.operands[0].name)
                return f"goto {label};"
            elif len(instr.operands) == 3:  # Conditional
                cond = instr.operands[0].name
                true_label = block_labels.get(instr.operands[1].name, instr.operands[1].name)
                false_label = block_labels.get(instr.operands[2].name, instr.operands[2].name)
                return f"if ({cond}) goto {true_label}; else goto {false_label};"

        # Memory operations
        elif opcode == "alloca":
            type_name = self._convert_llvm_type(instr.type.pointee)
            return f"{type_name} {instr.name};"
        elif opcode == "load" and len(instr.operands) >= 1:
            return f"{instr.name} = *{instr.operands[0].name};"
        elif opcode == "store" and len(instr.operands) >= 2:
            return f"*{instr.operands[1].name} = {instr.operands[0].name};"
        elif opcode == "getelementptr" and len(instr.operands) >= 2:
            base_ptr = instr.operands[0].name
            indices = [op.name for op in instr.operands[1:]]
            index_expr = " + ".join(indices) if indices else "0"
            return f"{instr.name} = {base_ptr} + {index_expr};"

        # Function calls
        elif opcode == "call" and len(instr.operands) >= 1:
            func_name = instr.operands[-1].name
            args = [op.name for op in instr.operands[:-1]]
            return (f"{instr.name} = {func_name}({', '.join(args)});" 
                   if str(instr.type) != "void" else f"{func_name}({', '.join(args)});")

        # Type conversions
        elif opcode in ["trunc", "zext", "sext", "fptrunc", "fpext", "bitcast", 
                      "inttoptr", "ptrtoint"] and len(instr.operands) >= 1:
            target_type = self._convert_llvm_type(instr.type)
            return f"{instr.name} = ({target_type}){instr.operands[0].name};"

        # PHI nodes
        elif opcode == "phi" and len(instr.operands) >= 2:
            val = instr.operands[0].name
            return f"{instr.name} = {val}; /* Simplified PHI node */"

        return f"/* Unsupported: {opcode} */"

    def compile_and_load_ptx(self, ptx_code: PTXCode, function_name: str) -> KernelFunction:
        """Compile PTX code and load the specified function.
        
        This method compiles PTX assembly code and loads a specific function
        as a CUDA kernel.
        
        Args:
            ptx_code: PTX assembly code string
            function_name: Name of function to load
            
        Returns:
            CUDA kernel function
            
        Raises:
            Exception: If compilation fails
            
        The method caches compiled kernels to avoid recompilation.
        """
        if function_name in self.kernel_cache:
            return self.kernel_cache[function_name]

        try:
            module = SourceModule(ptx_code, no_extern_c=True)
            kernel = module.get_function(function_name)
            self.kernel_cache[function_name] = kernel
            return kernel
        except Exception as e:
            logger.error(f"Failed to compile PTX: {e}")
            raise

    def execute_kernel(self, kernel_func: KernelFunction, *args: Any, 
                      grid: Tuple[int, int, int] = (1,1,1), 
                      block: Tuple[int, int, int] = (32,1,1)) -> None:
        """Execute a CUDA kernel with the specified arguments.
        
        Args:
            kernel_func: CUDA kernel function
            *args: Kernel arguments
            grid: Grid dimensions
            block: Block dimensions
            
        Raises:
            Exception: If kernel execution fails
        """
        try:
            kernel_func(*args, grid=grid, block=block)
            cuda.Context.synchronize()
        except Exception as e:
            logger.error(f"Kernel execution failed: {e}")
            raise

    def end_to_end_execute(self, llvm_ir: str, 
                          input_data: Optional[Union[int, np.ndarray]] = None,
                          output_type: np.dtype = np.int32) -> Tuple[Optional[Union[int, np.ndarray]], float]:
        """Execute LLVM IR end-to-end on GPU.
        
        Args:
            llvm_ir: LLVM IR code string
            input_data: Input data (integer or numpy array)
            output_type: Output data type
            
        Returns:
            Tuple containing:
                - Execution result (None for void functions)
                - Execution time in seconds
                
        Raises:
            ValueError: If no kernel function found
            TypeError: If input data type is not supported
            Exception: If execution fails
        """
        try:
            # Convert to PTX
            ptx_code = self.llvm_to_ptx(llvm_ir)
            
            # Get kernel name
            mod = self.parse_module(llvm_ir)
            kernel_name = next((f.name for f in mod.functions if not f.is_declaration), None)
            if not kernel_name:
                raise ValueError("No kernel function found")

            # Compile and load kernel
            kernel_func = self.compile_and_load_ptx(ptx_code, kernel_name)
            
            # Prepare data
            if input_data is not None:
                if isinstance(input_data, int):
                    d_input = cuda.to_device(np.array([input_data], dtype=np.int32))
                    d_output = cuda.mem_alloc(np.dtype(output_type).itemsize)
                elif isinstance(input_data, np.ndarray):
                    d_input = cuda.to_device(input_data)
                    d_output = cuda.mem_alloc(np.dtype(output_type).itemsize * 
                                             (input_data.size if output_type != np.void else 1))
                else:
                    raise TypeError(f"Unsupported input data type: {type(input_data)}")
            else:
                d_input = None
                d_output = cuda.mem_alloc(np.dtype(output_type).itemsize)
            
            # Execute kernel
            start_time = time.time()
            if d_input is not None:
                self.execute_kernel(
                    kernel_func, d_input, d_output, 
                    block=(256, 1, 1), grid=((input_data.size if isinstance(input_data, np.ndarray) else 1 + 255) // 256, 1, 1)
                )
            else:
                self.execute_kernel(
                    kernel_func, d_output,
                    block=(256, 1, 1), grid=(1, 1, 1)
                )
            execution_time = time.time() - start_time
            
            # Retrieve result
            if output_type == np.void:
                return None, execution_time
                
            if isinstance(input_data, np.ndarray):
                result = np.zeros(input_data.shape, dtype=output_type)
            else:
                result = np.zeros(1, dtype=output_type)
                
            cuda.memcpy_dtoh(result, d_output)
            
            return result[0] if result.size == 1 else result, execution_time
            
        except Exception as e:
            logger.error(f"End-to-end execution failed: {e}")
            raise
            
    def execute_cpu_for_validation(self, llvm_ir: str, 
                                 input_data: Optional[Union[int, np.ndarray]] = None) -> int:
        """Execute on CPU for validation purposes.
        
        Args:
            llvm_ir: LLVM IR code string
            input_data: Input data (integer or numpy array)
            
        Returns:
            Execution result
            
        Raises:
            ValueError: If no function found
            TypeError: If input type not supported
            Exception: If execution fails
        """
        try:
            import ctypes
            # Parse and optimize module
            mod = self.parse_module(llvm_ir)
            
            # Create execution engine
            target = llvm.Target.from_default_triple()
            target_machine = target.create_target_machine()
            engine = llvm.create_mcjit_compiler(mod, target_machine)
            
            # Get function
            func_name = next((f.name for f in mod.functions if not f.is_declaration), None)
            if not func_name:
                raise ValueError("No function found to execute")
            
            func_ptr = engine.get_function_address(func_name)
            
            # Create callable function based on input data
            if input_data is not None:
                if isinstance(input_data, int):
                    cfunc = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)(func_ptr)
                    result = cfunc(input_data)
                elif isinstance(input_data, np.ndarray):
                    cfunc = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)(func_ptr)
                    result = cfunc(input_data.ctypes.data)
                else:
                    raise TypeError(f"Unsupported input type: {type(input_data)}")
            else:
                cfunc = ctypes.CFUNCTYPE(ctypes.c_int)(func_ptr)
                result = cfunc()
                
            return result
        except Exception as e:
            logger.error(f"CPU execution failed: {e}")
            raise

if __name__ == "__main__":
    # サンプルコード
    llvm_ir = """
    define i32 @add(i32 %a) {
        %result = add i32 %a, 10
        ret i32 %result
    }
    """
    
    # 入力データ
    input_data = 5
    
    # CPUで実行
    executor = CPUExecutor()
    result = executor.execute(llvm_ir, input_data)
    print(f"入力: {input_data}")
    print(f"結果: {result}")  # 15が出力されるはず
