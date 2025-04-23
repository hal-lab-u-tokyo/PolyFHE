# In order to use type hint of self in the class, we need to import annotations
from __future__ import annotations

import uuid
import os
from graphviz import Digraph
from enum import Enum, auto

class PolyOpType(Enum):
    Add = auto()
    Sub = auto()
    Mult = auto()
    Decomp = auto()
    BConv = auto()
    ModDown = auto()
    ModUp = auto()
    NTT = auto()
    NTTPhase1 = auto()
    NTTPhase2 = auto()
    iNTT = auto()
    iNTTPhase1 = auto()
    iNTTPhase2 = auto()
    Init = auto()
    End = auto()
    # Special Edge
    Const = auto()
    InitEdge = auto()
    EndEdge = auto()

    def __str__(self):
        return self.name
    
    def __format__(self, format_spec):
        return self.name

class PolyOp:
    def __init__(self, op_type: PolyOpType, inputs: list[PolyOp], name: str, current_limb: int = 0, start_limb: int = 0, end_limb: int = 0) -> None:
        print(f"Creating PolyOp {op_type} with inputs {inputs}")
        self.id: str = str(uuid.uuid4())
        self.name: str = name
        self.op_type: PolyOpType = op_type
        self.inputs: list[PolyOp] = inputs
        self.current_limb: int = current_limb
        self.start_limb: int = start_limb
        self.end_limb: int = end_limb

    def __str__(self):
        return f"PolyOp(op:{self.op_type},id:{self.id[-4:]})"
    
    def __format__(self, format_spec):
        return f"PolyOp(op:{self.op_type},id:{self.id[-4:]})"
    
    def __repr__(self):
        return f"PolyOp(op:{self.op_type},id:{self.id[-4:]})"

    def set_special_edge(self, idx_ct: int, offset: int, n_poly: int = 0):
        assert(self.op_type == PolyOpType.InitEdge or self.op_type == PolyOpType.EndEdge or self.op_type == PolyOpType.Malloc)
        self.idx_ct = idx_ct
        self.offset = offset
        self.n_poly = n_poly

    def set_decomp_info(self, alpha: int, beta: int):
        assert(self.op_type == PolyOpType.Decomp)
        self.alpha = alpha
        self.beta = beta

class PolyFHE:
    def __init__(self):
        self.operations = []
        self.results = {}
        self.init_op = PolyOp(PolyOpType.Init, [], "init")
        self.end_op = PolyOp(PolyOpType.End, [], "end")

    def init(self, idx_ct: int, offset: int, name: str):
        op =  PolyOp(PolyOpType.InitEdge, [], name)
        op.set_special_edge(idx_ct, offset)
        return op
    
    def end(self, a: PolyOp, ct_idx: int, offset: int):
        op = PolyOp(PolyOpType.EndEdge, [a], "end")
        op.set_special_edge(ct_idx, offset)
        return op

    def add(self, a: PolyOp, b: PolyOp, name: str, current_limb: int, start_limb: int, end_limb: int):
        return PolyOp(PolyOpType.Add, [a, b], name, current_limb, start_limb, end_limb)
    
    def mul(self, a: PolyOp, b: PolyOp, name: str, current_limb: int, start_limb: int, end_limb: int):
        return PolyOp(PolyOpType.Mult, [a, b], name, current_limb, start_limb, end_limb)
    
    def decomp(self, a: PolyOp, name: str, current_limb: int, start_limb: int, end_limb: int, alpha: int, beta: int):
        op = PolyOp(PolyOpType.Decomp, [a], name, current_limb, start_limb, end_limb)
        op.set_decomp_info(alpha, beta)
        return op
    
    def intt_phase1(self, a: PolyOp, name: str, current_limb: int, start_limb: int, end_limb: int):
        return PolyOp(PolyOpType.iNTTPhase1, [a], name, current_limb, start_limb, end_limb)
    
    def intt_phase2(self, a: PolyOp, name: str, current_limb: int, start_limb: int, end_limb: int):
        return PolyOp(PolyOpType.iNTTPhase2, [a], name, current_limb, start_limb, end_limb)
    
    def compile(self, results: list[PolyOp], filename="build/output"):
        print("Compiling PolyFHE graph...")
        dot = Digraph(comment='PolyFHE Graph')
        dot.node(f"{self.init_op.name}", label=f"{self.init_op.op_type}")
        dot.node(f"{self.end_op.name}", label=f"{self.end_op.op_type}")


        def add_node(op: PolyOp):
            if op.op_type == PolyOpType.InitEdge or op.op_type == PolyOpType.EndEdge:
                return
            dot.node(f"{op.name}", label=f"{op.op_type}")

        def add_edge(src: PolyOp, dst: PolyOp):
            if src.op_type == PolyOpType.InitEdge:
                dot.edge(self.init_op.name, dst.name, label=gen_edge_label(src, dst))
            elif dst.op_type == PolyOpType.EndEdge:
                dot.edge(src.name, self.end_op.name, label=gen_edge_label(src, dst))
            else:
                dot.edge(src.name, dst.name, label=gen_edge_label(src, dst))

        def gen_edge_label(src: PolyOp, dst: PolyOp):
            print(f"Generating edge label for {src} -> {dst}")  
            label = ""
            if src.op_type == PolyOpType.InitEdge:
                label = f"{dst.end_limb}_init_{src.idx_ct}_{src.offset}"
            elif dst.op_type == PolyOpType.EndEdge:
                label = f"{src.end_limb}_end_{dst.idx_ct}_{dst.offset}"
            else:
                if src.current_limb != dst.current_limb:
                    print(f"error: current limb mismatch {src.current_limb} != {dst.current_limb}")
                    exit(1)
                if src.start_limb != dst.start_limb:
                    print(f"error: start limb mismatch {src.start_limb} != {dst.start_limb}")
                    exit(1)
                if src.end_limb != dst.end_limb:
                    print(f"error: end limb mismatch {src.end_limb} != {dst.end_limb}")
                    exit(1)
                label = f"{src.end_limb}"
            return label

        def gen_dot(op: PolyOp, visited: set):
            if op.id in visited:
                return

            print(f"Visiting {op}, input: {op.inputs}")
            visited.add(op.id)

            # Add the node to the graph
            add_node(op)

            # Add input edges
            for next in op.inputs:
                if len(next.inputs) == 0:
                    add_node(next)
                    add_edge(next, op)
                else:
                    add_edge(next, op)
                    gen_dot(next, visited)

        visited = set()
        for result in results:
            gen_dot(result, visited)
        
        # Export to current directory
        dot.render(filename, format='dot', cleanup=True)
        dot.render(filename, format='png', cleanup=True)
        print(f"Graph compiled and saved to {filename}")
