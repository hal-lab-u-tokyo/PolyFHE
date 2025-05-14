# In order to use type hint of self in the class, we need to import annotations
from __future__ import annotations

import uuid
import os
from graphviz import Digraph
from enum import Enum, auto


class PolyOpType(Enum):
    Add = auto()
    Accum = auto()
    Sub = auto()
    Mult = auto()
    MultConst = auto()
    MultKey = auto()
    MultKeyAccum = auto()
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
    def __init__(
        self,
        op_type: PolyOpType,
        inputs: list[PolyOp],
        name: str,
        in_start_limb: int = 0,
        in_end_limb: int = 0,
        out_start_limb: int = 0,
        out_end_limb: int = 0,
    ) -> None:
        print(f"Creating PolyOp {op_type} with inputs {inputs}")
        self.id: str = str(uuid.uuid4())
        self.name: str = name
        self.op_type: PolyOpType = op_type
        self.inputs: list[PolyOp] = inputs
        self.in_start_limb: int = in_start_limb
        self.in_end_limb: int = in_end_limb
        self.out_start_limb: int = out_start_limb
        self.out_end_limb: int = out_end_limb

    def __str__(self):
        return f"PolyOp(op:{self.op_type},id:{self.id[-4:]})"

    def __format__(self, format_spec):
        return f"PolyOp(op:{self.op_type},id:{self.id[-4:]})"

    def __repr__(self):
        return f"PolyOp(op:{self.op_type},id:{self.id[-4:]})"

    def set_special_edge(self, idx_ct: int, offset: int, n_poly: int = 0):
        assert (
            self.op_type == PolyOpType.InitEdge
            or self.op_type == PolyOpType.EndEdge
            or self.op_type == PolyOpType.Malloc
        )
        self.idx_ct = idx_ct
        self.offset = offset
        self.n_poly = n_poly

    def set_bconv_info(self, beta_idx: int):
        assert self.op_type == PolyOpType.BConv
        self.beta_idx = beta_idx

class MulKeyAccumOp(PolyOp):
    def __init__(
        self,
        inputs: list[PolyOp],
        name: str,
        start_limb: int,
        end_limb: int,
        beta: int,
    ) -> None:
        super().__init__(PolyOpType.MultKeyAccum, inputs, name, start_limb, end_limb, start_limb, end_limb)
        self.beta: int = beta

class NTTOp(PolyOp):
    def __init__(
        self,
        op_type: PolyOpType,
        inputs: list[PolyOp],
        name: str,
        start_limb: int,
        end_limb: int,
        exclude_start_limb: int,
        exclude_end_limb: int,
    ) -> None:
        assert (
            op_type == PolyOpType.NTTPhase1
            or op_type == PolyOpType.NTTPhase2
            or op_type == PolyOpType.iNTTPhase1
            or op_type == PolyOpType.iNTTPhase2
        )
        super().__init__(
            op_type, inputs, name, start_limb, end_limb, start_limb, end_limb
        )
        self.exclude_start_limb: int = exclude_start_limb
        self.exclude_end_limb: int = exclude_end_limb


class PolyFHE:
    def __init__(self):
        self.operations = []
        self.results = {}
        self.init_op = PolyOp(PolyOpType.Init, [], "init")
        self.end_op = PolyOp(PolyOpType.End, [], "end")

    def init(self, name: str, idx_ct: int, offset: int):
        op = PolyOp(PolyOpType.InitEdge, [], name)
        op.set_special_edge(idx_ct, offset)
        return op

    def end(self, a: PolyOp, ct_idx: int, offset: int):
        op = PolyOp(PolyOpType.EndEdge, [a], "end")
        op.set_special_edge(ct_idx, offset)
        return op

    def add(self, a: PolyOp, b: PolyOp, name: str, start_limb: int, end_limb: int):
        return PolyOp(
            PolyOpType.Add, [a, b], name, start_limb, end_limb, start_limb, end_limb
        )

    def accum(self, a: [PolyOp], name: str, start_limb: int, end_limb: int):
        return PolyOp(
            PolyOpType.Accum, a, name, start_limb, end_limb, start_limb, end_limb
        )

    def mul(self, a: PolyOp, b: PolyOp, name: str, start_limb: int, end_limb: int):
        return PolyOp(
            PolyOpType.Mult, [a, b], name, start_limb, end_limb, start_limb, end_limb
        )

    def mul_const(self, a: PolyOp, name: str, start_limb: int, end_limb: int):
        return PolyOp(
            PolyOpType.MultConst, [a], name, start_limb, end_limb, start_limb, end_limb
        )

    def mul_key(self, a: PolyOp, name: str, start_limb: int, end_limb: int):
        return PolyOp(
            PolyOpType.MultKey, [a], name, start_limb, end_limb, start_limb, end_limb
        )

    def mul_key_accum(self, a: [PolyOp], name: str, start_limb: int, end_limb: int, beta: int):
        return MulKeyAccumOp(
            a, name, start_limb, end_limb, beta
        )

    def bconv(self, a: PolyOp, name: str, current_limb: int, beta_idx: int, alpha: int):
        # op = PolyOp(PolyOpType.BConv, [a], name, beta_idx * alpha, (beta_idx + 1) * alpha, 0, current_limb + alpha)
        op = PolyOp(
            PolyOpType.BConv, [a], name, 0, current_limb, 0, current_limb + alpha
        )
        op.set_bconv_info(beta_idx)
        return op

    def ntt(
        self,
        a: PolyOp,
        name: str,
        if_forward: bool,
        if_phase1: bool,
        start_limb: int,
        end_limb: int,
        exclude_start: int,
        exclude_end: int,
    ):
        op_type = None
        if if_forward:
            if if_phase1:
                op_type = PolyOpType.NTTPhase1
            else:
                op_type = PolyOpType.NTTPhase2
        else:
            if if_phase1:
                op_type = PolyOpType.iNTTPhase1
            else:
                op_type = PolyOpType.iNTTPhase2
        return NTTOp(
            op_type,
            [a],
            name,
            start_limb,
            end_limb,
            exclude_start,
            exclude_end,
        )

    def ntt_phase1(
        self,
        a: PolyOp,
        name: str,
        start_limb: int,
        end_limb: int,
        exclude_start: int,
        exclude_end: int,
    ):
        return NTTOp(
            PolyOpType.NTTPhase1,
            [a],
            name,
            start_limb,
            end_limb,
            exclude_start,
            exclude_end,
        )

    def ntt_phase2(
        self,
        a: PolyOp,
        name: str,
        start_limb: int,
        end_limb: int,
        exclude_start: int,
        exclude_end: int,
    ):
        return NTTOp(
            PolyOpType.NTTPhase2,
            [a],
            name,
            start_limb,
            end_limb,
            exclude_start,
            exclude_end,
        )

    def intt_phase1(self, a: PolyOp, name: str, start_limb: int, end_limb: int):
        return PolyOp(
            PolyOpType.iNTTPhase1, [a], name, start_limb, end_limb, start_limb, end_limb
        )

    def intt_phase2(self, a: PolyOp, name: str, start_limb: int, end_limb: int):
        return PolyOp(
            PolyOpType.iNTTPhase2, [a], name, start_limb, end_limb, start_limb, end_limb
        )

    def compile(self, results: list[PolyOp], filename="build/output"):
        print("Compiling PolyFHE graph...")
        dot = Digraph(comment="PolyFHE Graph")
        dot.node(f"{self.init_op.name}", label=f"{self.init_op.op_type}")
        dot.node(f"{self.end_op.name}", label=f"{self.end_op.op_type}")

        def gen_edge_label(src: PolyOp, dst: PolyOp):
            print(f"Generating edge label for {src} -> {dst}")
            label = ""
            if src.op_type == PolyOpType.InitEdge:
                label = f"{dst.in_end_limb}_init_{src.idx_ct}_{src.offset}"
            elif dst.op_type == PolyOpType.EndEdge:
                label = f"{src.out_end_limb}_end_{dst.idx_ct}_{dst.offset}"
            else:
                label = f"{dst.in_end_limb - dst.in_start_limb}"
            return label

        def gen_node_label(op: PolyOp):
            label = f"{op.op_type}"
            if op.op_type == PolyOpType.BConv:
                label += f"_{op.beta_idx}"
            elif isinstance(op, MulKeyAccumOp):
                label += f"_{op.in_start_limb}_{op.in_end_limb}_{op.beta}"
            elif isinstance(op, NTTOp):
                label += f"_{op.in_start_limb}_{op.in_end_limb}_{op.exclude_start_limb}_{op.exclude_end_limb}"
            else:
                label += f"_{op.in_start_limb}_{op.in_end_limb}"
            return label

        def add_node(op: PolyOp):
            if op.op_type == PolyOpType.InitEdge or op.op_type == PolyOpType.EndEdge:
                return
            dot.node(f"{op.name}", label=f"{gen_node_label(op)}")

        def add_edge(src: PolyOp, dst: PolyOp):
            if src.op_type == PolyOpType.InitEdge:
                dot.edge(self.init_op.name, dst.name, label=gen_edge_label(src, dst))
            elif dst.op_type == PolyOpType.EndEdge:
                dot.edge(src.name, self.end_op.name, label=gen_edge_label(src, dst))
            else:
                dot.edge(src.name, dst.name, label=gen_edge_label(src, dst))

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
        dot.render(filename, format="dot", cleanup=True)
        dot.render(filename, format="png", cleanup=True)
        print(f"Graph compiled and saved to {filename}")
