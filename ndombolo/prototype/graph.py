"""
The static script graph (paper, Prop. 2.4).

One traversal of the parse tree with a scope stack, emitting every item and
every contact exactly once. No execution is involved: the graph exists before
any run, which is the sense in which the script *is* the causal knowledge
graph rather than describing one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

Node = Dict[str, Any]

# Declaration forms that introduce a named binding site (paper, Def. 2.2).
DECL_KINDS = {
    "Item": "item",
    "Funxn": "funxn",
    "Proposition": "proposition",
    "Motion": "motion",
    "Point": "point",
    "Hypothesis": "hypothesis",
}


@dataclass
class ScriptGraph:
    """Items, weighted contacts, and the stage each item was declared at."""

    items: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    contacts: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def declare(self, name: str, kind: str, line: int, stage: int) -> None:
        # A redeclaration keeps the first site: items are not undeclared, and
        # the earliest declaration is the one later stages contain.
        self.items.setdefault(name, {
            "name": name, "kind": kind, "line": line, "stage": stage,
        })

    def contact(self, src: str, dst: str) -> None:
        if src == dst or src not in self.items or dst not in self.items:
            return
        key = (src, dst)
        self.contacts[key] = self.contacts.get(key, 0) + 1

    def to_json(self) -> Dict[str, Any]:
        return {
            "items": sorted(self.items.values(), key=lambda d: (d["stage"], d["line"], d["name"])),
            "contacts": [
                {"from": a, "to": b, "weight": w}
                for (a, b), w in sorted(self.contacts.items())
            ],
            "item_count": len(self.items),
            "contact_count": len(self.contacts),
        }


class GraphBuilder:
    """Accretive builder: `add_cell` extends the graph, never rewrites it.

    Prefix containment (paper, Prop. 2.7) holds because no method removes an
    item or decreases a weight.
    """

    def __init__(self) -> None:
        self.graph = ScriptGraph()
        self._stage = 0

    def add_cell(self, body: List[Node], stage: int) -> None:
        self._stage = stage
        # Two passes per cell: declare, then link. A statement may mention an
        # item declared later in the same cell.
        self._declare(body)
        self._link(body, opener=None)

    # -- pass 1: declarations --------------------------------------------

    def _declare(self, body: List[Node]) -> None:
        for st in body:
            kind = DECL_KINDS.get(st["kind"])
            if kind is not None:
                self.graph.declare(st["name"], kind, st["line"], self._stage)
            for child in _child_blocks(st):
                self._declare(child)

    # -- pass 2: contacts -------------------------------------------------

    def _link(self, body: List[Node], opener: str | None) -> None:
        for st in body:
            self._link_stmt(st, opener)

    def _link_stmt(self, st: Node, opener: str | None) -> None:
        kind = st["kind"]
        # A declaration is the target of contacts from what its term mentions.
        target = st.get("name") if kind in DECL_KINDS else opener

        for expr in _exprs_of(st):
            for name in free_names(expr):
                if target is not None:
                    self.graph.contact(name, target)
                elif opener is not None:
                    self.graph.contact(name, opener)

        # A verdict contacts the motion it is about.
        if kind in ("Support", "Contradict", "Inconclusive"):
            motion = st["motion"]
            if target is not None:
                self.graph.contact(motion, target)
            elif opener is not None:
                self.graph.contact(motion, opener)

        # A declaration with a body opens a scope: everything inside is in
        # contact with it (paper, Def. 2.3, second clause).
        inner_opener = st["name"] if kind in DECL_KINDS else opener
        if kind == "Proposition":
            for m in st.get("motions", []):
                self.graph.contact(st["name"], m["name"])
        for child in _child_blocks(st):
            self._link(child, inner_opener)


# -- AST walking ---------------------------------------------------------

_BLOCK_FIELDS = ("body", "then", "otherwise", "motions")


def _child_blocks(st: Node) -> List[List[Node]]:
    out: List[List[Node]] = []
    for f in _BLOCK_FIELDS:
        v = st.get(f)
        if isinstance(v, list) and v and isinstance(v[0], dict) and "kind" in v[0]:
            out.append(v)
    return out


_EXPR_FIELDS = ("expr", "cond", "iter", "target", "conf")


def _exprs_of(st: Node) -> List[Node]:
    return [st[f] for f in _EXPR_FIELDS if isinstance(st.get(f), dict)]


def free_names(expr: Node) -> Set[str]:
    """Names occurring free in an expression."""
    out: Set[str] = set()
    _walk_expr(expr, out)
    return out


def _walk_expr(e: Any, out: Set[str]) -> None:
    if not isinstance(e, dict):
        return
    kind = e.get("kind")
    if kind == "Var":
        out.add(e["name"])
        return
    if kind == "Call":
        _walk_expr(e["callee"], out)
        for a in e["args"]:
            _walk_expr(a, out)
        return
    if kind == "Map":
        for entry in e["entries"]:
            _walk_expr(entry["value"], out)
        return
    if kind == "List":
        for it in e["items"]:
            _walk_expr(it, out)
        return
    for key in ("left", "right", "operand", "target", "index"):
        if key in e:
            _walk_expr(e[key], out)
