# In order to use type hint of self in the class, we need to import annotations
from __future__ import annotations

import uuid
from graphviz import Digraph

class PolyOp:
    def __init__(self, op_type: str, inputs: list[PolyOp]) -> None:
        self.id: str = str(uuid.uuid4())
        self.op_type: str = op_type
        self.inputs: list[PolyOp] = inputs

    def __init__(self, op_type: str, input: int) -> None:
        self.id: str = str(uuid.uuid4())
        self.op_type: str = op_type
        self.inputs: list[PolyOp] = [input]


class PolyFHE:
    def __init__(self):
        self.operations = []
        self.results = {}

    def add(self, a: Union[PolyOp,int], b: Union[PolyOp,int]):
        if not isinstance(a, PolyOp):
            a = PolyOp("Const", [a])
        return PolyOp("Add", [a, b])
    
    def sub(self, a: PolyOp, b: PolyOp):
        return PolyOp("Sub", [a, b])
    
    def mul(self, a, b):
        return PolyOp("Mul", [a, b])
    
    def compile(self, result, filename="data/output"):
        print("Compiling PolyFHE graph...")
        dot = Digraph(comment='PolyFHE Graph')

        visited = set()

        def add_node(op):
            if op.id in visited:
                return
            visited.add(op.id)

            # Add the node to the graph
            dot.node(op.id, label=f"{op.op_type}({', '.join(map(str, op.inputs))})")

            # Add input edges
            for input_op in op.inputs:
                if isinstance(input_op, PolyOp):
                    add_node(input_op)
                    dot.edge(input_op.id, op.id)
                else:
                    const_id = f"const_{hash(input_op)}"
                    dot.node(const_id, label=str(input_op))
                    dot.edge(const_id, op.id)
        
        add_node(result)
        dot.render(filename, format='dot', cleanup=True)
        dot.render(filename, format='png', cleanup=True)
        print(f"Graph compiled and saved to {filename}")
