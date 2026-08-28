"""
The deterministic compiler K (paper, Def. 3.2).

Big-step evaluation of the deterministic Turbulance fragment. Every rule
application emits one trace event recording rule, site, items read with their
values, and the item written with its value -- which is what makes every
atomic question a projection of the trace (paper, Thm. 5.2).

No rule consults a clock, an address, a random source, or a model
(paper, Premise 1). That is checkable by reading this file: the only inputs to
`exec_stmt` and `eval_expr` are the AST node and the environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graph import free_names

Node = Dict[str, Any]

MAX_STEPS = 200_000       # guards non-termination; exceeding it is an error
MAX_LOOP = 100_000


class RuntimeErr(Exception):
    def __init__(self, message: str, line: int):
        super().__init__(f"line {line}: {message}")
        self.message = message
        self.line = line


class ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value


# -- values --------------------------------------------------------------


@dataclass
class Closure:
    name: str
    params: List[str]
    body: List[Node]
    env: "Env"


@dataclass
class Point:
    name: str
    content: Dict[str, Any]
    confidence: float


@dataclass
class Motion:
    name: str
    text: str


def tag(v: Any) -> str:
    """The runtime value tag, mirroring the reference implementation."""
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, float) or isinstance(v, int):
        return "num"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, Point):
        return "point"
    if isinstance(v, Motion):
        return "motion"
    if isinstance(v, Closure) or callable(v):
        return "closure"
    if isinstance(v, dict):
        return "map"
    return "unknown"


def render(v: Any) -> Any:
    """A JSON-safe rendering of a runtime value."""
    if isinstance(v, Closure):
        return {"closure": v.name, "params": v.params}
    if callable(v):
        return {"builtin": getattr(v, "__name__", "builtin")}
    if isinstance(v, Point):
        return {"point": v.name, "content": {k: render(x) for k, x in v.content.items()},
                "confidence": v.confidence}
    if isinstance(v, Motion):
        return {"motion": v.name, "text": v.text}
    if isinstance(v, list):
        return [render(x) for x in v]
    if isinstance(v, dict):
        return {k: render(x) for k, x in v.items()}
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


# -- environment ---------------------------------------------------------


class Env:
    def __init__(self, parent: Optional["Env"] = None):
        self.vars: Dict[str, Any] = {}
        self.parent = parent

    def get(self, name: str, line: int) -> Any:
        env: Optional[Env] = self
        while env is not None:
            if name in env.vars:
                return env.vars[name]
            env = env.parent
        raise RuntimeErr(f"undefined name {name!r}", line)

    def has(self, name: str) -> bool:
        env: Optional[Env] = self
        while env is not None:
            if name in env.vars:
                return True
            env = env.parent
        return False

    def define(self, name: str, value: Any) -> None:
        self.vars[name] = value

    def assign(self, name: str, value: Any, line: int) -> None:
        env: Optional[Env] = self
        while env is not None:
            if name in env.vars:
                env.vars[name] = value
                return
            env = env.parent
        raise RuntimeErr(f"assignment to undefined name {name!r}", line)

    def flatten(self) -> Dict[str, Any]:
        """The visible store, innermost binding winning."""
        out: Dict[str, Any] = {}
        chain: List[Env] = []
        env: Optional[Env] = self
        while env is not None:
            chain.append(env)
            env = env.parent
        for e in reversed(chain):
            out.update(e.vars)
        return out


# -- the compiler --------------------------------------------------------


@dataclass
class Verdict:
    motion: str
    stance: str          # support | contradict | inconclusive
    confidence: float
    line: int


@dataclass
class PropositionState:
    name: str
    motions: List[Motion] = field(default_factory=list)
    verdicts: List[Verdict] = field(default_factory=list)

    def score(self, motion: str) -> float:
        """Noisy-or aggregation over verdicts for one motion."""
        pos = [v.confidence for v in self.verdicts
               if v.motion == motion and v.stance == "support"]
        neg = [v.confidence for v in self.verdicts
               if v.motion == motion and v.stance == "contradict"]
        return _noisy_or(pos) * (1.0 - _noisy_or(neg))


def _noisy_or(cs: List[float]) -> float:
    acc = 1.0
    for c in cs:
        acc *= (1.0 - _clamp(c))
    return 1.0 - acc


def _clamp(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


class Compiler:
    """K, carried across cells: the store and the trace both accrete."""

    def __init__(self) -> None:
        self.env = Env()
        self.trace: List[Dict[str, Any]] = []
        self.output: List[str] = []
        self.propositions: Dict[str, PropositionState] = {}
        self.points: Dict[str, Point] = {}
        self.steps = 0
        self._prop_stack: List[PropositionState] = []
        self._install_builtins()

    # -- tracing ---------------------------------------------------------

    def emit(self, rule: str, line: int, reads: Dict[str, Any],
             writes: Optional[str] = None, value: Any = None,
             note: Optional[str] = None) -> None:
        event: Dict[str, Any] = {
            "seq": len(self.trace),
            "rule": rule,
            "site": {"line": line},
            "reads": [{"item": k, "value": render(v)} for k, v in reads.items()],
        }
        if writes is not None:
            event["writes"] = {"item": writes, "value": render(value),
                               "tag": tag(value)}
        if note is not None:
            event["note"] = note
        self.trace.append(event)

    def _reads_for(self, expr: Optional[Node]) -> Dict[str, Any]:
        """The bound names an expression mentions, with their current values."""
        if expr is None:
            return {}
        out: Dict[str, Any] = {}
        for name in sorted(free_names(expr)):
            if self.env.has(name):
                out[name] = self.env.get(name, expr.get("line", 0))
        return out

    def _tick(self, line: int) -> None:
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise RuntimeErr("step limit exceeded", line)

    # -- statements -------------------------------------------------------

    def exec_block(self, body: List[Node]) -> None:
        for st in body:
            self.exec_stmt(st)

    def exec_stmt(self, st: Node) -> None:
        self._tick(st.get("line", 0))
        handler = getattr(self, f"_st_{st['kind']}", None)
        if handler is None:
            raise RuntimeErr(f"no rule for {st['kind']}", st.get("line", 0))
        handler(st)

    def _st_Noop(self, st: Node) -> None:
        self.emit("noop", st["line"], {}, note=st.get("text", ""))

    def _st_Funxn(self, st: Node) -> None:
        clo = Closure(st["name"], st["params"], st["body"], self.env)
        self.env.define(st["name"], clo)
        self.emit("declare-funxn", st["line"], {}, st["name"], clo)

    def _st_Item(self, st: Node) -> None:
        reads = self._reads_for(st["expr"])
        value = self.eval_expr(st["expr"])
        self.env.define(st["name"], value)
        self.emit("declare-item", st["line"], reads, st["name"], value)

    def _st_Assign(self, st: Node) -> None:
        reads = self._reads_for(st["expr"])
        value = self.eval_expr(st["expr"])
        if self.env.has(st["name"]):
            self.env.assign(st["name"], value, st["line"])
        else:
            self.env.define(st["name"], value)
        self.emit("assign", st["line"], reads, st["name"], value)

    def _st_Point(self, st: Node) -> None:
        reads = self._reads_for(st["expr"])
        raw = self.eval_expr(st["expr"])
        if not isinstance(raw, dict):
            raise RuntimeErr("point requires a map literal", st["line"])
        conf = _clamp(float(raw.get("certainty", raw.get("confidence", 1.0))))
        pt = Point(st["name"], raw, conf)
        self.env.define(st["name"], pt)
        self.points[st["name"]] = pt
        self.emit("declare-point", st["line"], reads, st["name"], pt)

    def _st_Hypothesis(self, st: Node) -> None:
        self.emit("declare-hypothesis", st["line"], {}, st["name"], st["name"])
        self.exec_block(st["body"])

    def _st_Proposition(self, st: Node) -> None:
        state = PropositionState(st["name"])
        for m in st["motions"]:
            motion = Motion(m["name"], m["text"])
            state.motions.append(motion)
            self.env.define(m["name"], motion)
            self.emit("declare-motion", m["line"], {}, m["name"], motion)
        self.propositions[st["name"]] = state
        self.emit("declare-proposition", st["line"], {}, st["name"],
                  [m.name for m in state.motions])

        self._prop_stack.append(state)
        try:
            self.exec_block(st["body"])
        finally:
            self._prop_stack.pop()

        for m in state.motions:
            self.emit("resolve-motion", st["line"],
                      {v.motion: v.confidence for v in state.verdicts
                       if v.motion == m.name},
                      f"{st['name']}.{m.name}", state.score(m.name))

    def _st_Motion(self, st: Node) -> None:
        motion = Motion(st["name"], st["text"])
        self.env.define(st["name"], motion)
        self.emit("declare-motion", st["line"], {}, st["name"], motion)

    def _st_Support(self, st: Node) -> None:
        self._verdict(st, "support")

    def _st_Contradict(self, st: Node) -> None:
        self._verdict(st, "contradict")

    def _st_Inconclusive(self, st: Node) -> None:
        self._verdict(st, "inconclusive")

    def _verdict(self, st: Node, stance: str) -> None:
        conf = 1.0
        reads: Dict[str, Any] = {}
        if st.get("conf") is not None:
            reads = self._reads_for(st["conf"])
            conf = _clamp(float(self.eval_expr(st["conf"])))
        if stance == "inconclusive":
            conf = 0.0
        if not self._prop_stack:
            raise RuntimeErr(f"{stance} outside a proposition", st["line"])
        state = self._prop_stack[-1]
        state.verdicts.append(Verdict(st["motion"], stance, conf, st["line"]))
        self.emit(stance, st["line"], reads,
                  f"{state.name}.{st['motion']}", conf)

    def _st_Given(self, st: Node) -> None:
        reads = self._reads_for(st["cond"])
        cond = truthy(self.eval_expr(st["cond"]))
        self.emit("given", st["line"], reads, note="then" if cond else "otherwise")
        self.exec_block(st["then"] if cond else st["otherwise"])

    def _st_Within(self, st: Node) -> None:
        reads = self._reads_for(st["target"])
        target = self.eval_expr(st["target"])
        self.emit("within", st["line"], reads)
        inner = Env(self.env)
        if isinstance(target, dict):
            for k, v in target.items():
                inner.define(k, v)
        saved, self.env = self.env, inner
        try:
            self.exec_block(st["body"])
        finally:
            self.env = saved

    def _st_Considering(self, st: Node) -> None:
        self._iterate(st, "considering")

    def _st_ForEach(self, st: Node) -> None:
        self._iterate(st, "for-each")

    def _iterate(self, st: Node, rule: str) -> None:
        reads = self._reads_for(st["iter"])
        seq = self.eval_expr(st["iter"])
        if not isinstance(seq, list):
            seq = [seq]
        if st.get("quant") == "item":
            seq = seq[:1]
        self.emit(rule, st["line"], reads, note=f"{len(seq)} iterations")
        for value in seq:
            inner = Env(self.env)
            inner.define(st["var"], value)
            saved, self.env = self.env, inner
            try:
                self.emit("bind", st["line"], {}, st["var"], value)
                self.exec_block(st["body"])
            finally:
                self.env = saved

    def _st_While(self, st: Node) -> None:
        rounds = 0
        while truthy(self.eval_expr(st["cond"])):
            rounds += 1
            if rounds > MAX_LOOP:
                raise RuntimeErr("loop limit exceeded", st["line"])
            self.emit("while", st["line"], self._reads_for(st["cond"]),
                      note=f"round {rounds}")
            self.exec_block(st["body"])
        self.emit("while-exit", st["line"], {}, note=f"{rounds} rounds")

    def _st_Return(self, st: Node) -> None:
        reads = self._reads_for(st.get("expr"))
        value = self.eval_expr(st["expr"]) if st.get("expr") else None
        self.emit("return", st["line"], reads, note=str(render(value)))
        raise ReturnSignal(value)

    def _st_Ensure(self, st: Node) -> None:
        reads = self._reads_for(st["expr"])
        ok = truthy(self.eval_expr(st["expr"]))
        self.emit("ensure", st["line"], reads, note="held" if ok else "failed")
        if not ok:
            raise RuntimeErr("ensure failed", st["line"])

    def _st_Resolve(self, st: Node) -> None:
        reads = self._reads_for(st["expr"])
        value = self.eval_expr(st["expr"])
        self.emit("resolve", st["line"], reads, note=str(render(value)))

    def _st_ExprStmt(self, st: Node) -> None:
        reads = self._reads_for(st["expr"])
        value = self.eval_expr(st["expr"])
        self.emit("expr", st["line"], reads, note=str(render(value)))

    # -- expressions ------------------------------------------------------

    def eval_expr(self, e: Node) -> Any:
        self._tick(e.get("line", 0))
        kind = e["kind"]

        if kind == "Num":
            return e["value"]
        if kind in ("Str", "Bool"):
            return e["value"]
        if kind == "None":
            return None
        if kind == "Var":
            return self.env.get(e["name"], e["line"])
        if kind == "List":
            return [self.eval_expr(x) for x in e["items"]]
        if kind == "Map":
            return {en["key"]: self.eval_expr(en["value"]) for en in e["entries"]}
        if kind == "Unary":
            v = self.eval_expr(e["operand"])
            return (not truthy(v)) if e["op"] == "not" else -_num(v, e["line"])
        if kind == "Binary":
            return self._binary(e)
        if kind == "Index":
            return self._index(e)
        if kind == "Field":
            return self._field(e)
        if kind == "Call":
            return self._call(e)
        if kind == "Pipe":
            # x |> f  is  f(x)
            return self._apply(self.eval_expr(e["right"]),
                               [self.eval_expr(e["left"])], e["line"])

        raise RuntimeErr(f"no rule for expression {kind}", e.get("line", 0))

    def _binary(self, e: Node) -> Any:
        op = e["op"]
        # Short-circuit: the right operand is not evaluated when it cannot matter.
        if op == "and":
            return truthy(self.eval_expr(e["left"])) and truthy(self.eval_expr(e["right"]))
        if op == "or":
            return truthy(self.eval_expr(e["left"])) or truthy(self.eval_expr(e["right"]))

        a, b = self.eval_expr(e["left"]), self.eval_expr(e["right"])
        line = e["line"]

        if op == "==":
            return render(a) == render(b)
        if op == "!=":
            return render(a) != render(b)
        if op == "+":
            if isinstance(a, str) or isinstance(b, str):
                return _text(a) + _text(b)
            if isinstance(a, list) and isinstance(b, list):
                return a + b
            return _num(a, line) + _num(b, line)
        if op == "-":
            return _num(a, line) - _num(b, line)
        if op == "*":
            return _num(a, line) * _num(b, line)
        if op == "/":
            d = _num(b, line)
            if d == 0.0:
                raise RuntimeErr("division by zero", line)
            return _num(a, line) / d
        if op == "%":
            d = _num(b, line)
            if d == 0.0:
                raise RuntimeErr("modulo by zero", line)
            return math.fmod(_num(a, line), d)
        if op in ("<", "<=", ">", ">="):
            x, y = _num(a, line), _num(b, line)
            return {"<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y}[op]

        raise RuntimeErr(f"unknown operator {op!r}", line)

    def _index(self, e: Node) -> Any:
        target = self.eval_expr(e["target"])
        idx = self.eval_expr(e["index"])
        if isinstance(target, list):
            i = int(_num(idx, e["line"]))
            if i < 0 or i >= len(target):
                raise RuntimeErr(f"index {i} out of range", e["line"])
            return target[i]
        if isinstance(target, dict):
            return target.get(_text(idx))
        raise RuntimeErr("value is not indexable", e["line"])

    def _field(self, e: Node) -> Any:
        target = self.eval_expr(e["target"])
        f = e["field"]
        if isinstance(target, Point):
            if f in ("confidence", "certainty"):
                return target.confidence
            return target.content.get(f)
        if isinstance(target, Motion):
            return {"name": target.name, "text": target.text}.get(f)
        if isinstance(target, dict):
            return target.get(f)
        raise RuntimeErr(f"no field {f!r} on {tag(target)}", e["line"])

    def _call(self, e: Node) -> Any:
        callee = self.eval_expr(e["callee"])
        args = [self.eval_expr(a) for a in e["args"]]
        return self._apply(callee, args, e["line"])

    def _apply(self, callee: Any, args: List[Any], line: int) -> Any:
        if callable(callee) and not isinstance(callee, Closure):
            return callee(args, line)
        if not isinstance(callee, Closure):
            raise RuntimeErr(f"{tag(callee)} is not callable", line)
        if len(args) != len(callee.params):
            raise RuntimeErr(
                f"{callee.name} expects {len(callee.params)} argument(s), "
                f"got {len(args)}", line)
        inner = Env(callee.env)
        for p, a in zip(callee.params, args):
            inner.define(p, a)
        saved, self.env = self.env, inner
        try:
            self.exec_block(callee.body)
            return None
        except ReturnSignal as r:
            return r.value
        finally:
            self.env = saved

    # -- builtins ---------------------------------------------------------

    def _install_builtins(self) -> None:
        def _print(args: List[Any], line: int) -> Any:
            text = " ".join(_text(a) for a in args)
            self.output.append(text)
            self.emit("print", line, {}, note=text)
            return None

        def _len(args: List[Any], line: int) -> Any:
            v = args[0] if args else None
            if isinstance(v, (list, str, dict)):
                return float(len(v))
            raise RuntimeErr(f"len of {tag(v)}", line)

        def _num_of(fn):
            def f(args: List[Any], line: int) -> Any:
                return fn([_num(a, line) for a in args], line)
            return f

        self.env.define("print", _print)
        self.env.define("len", _len)
        self.env.define("sum", _num_of(lambda xs, ln: float(sum(xs))))
        self.env.define("min", _num_of(lambda xs, ln: min(xs) if xs else 0.0))
        self.env.define("max", _num_of(lambda xs, ln: max(xs) if xs else 0.0))
        self.env.define("abs", _num_of(lambda xs, ln: abs(xs[0]) if xs else 0.0))
        self.env.define("round",
                        _num_of(lambda xs, ln: float(round(xs[0])) if xs else 0.0))


# -- helpers -------------------------------------------------------------


def truthy(v: Any) -> bool:
    if v is None or v is False:
        return False
    if v is True:
        return True
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, (str, list, dict)):
        return len(v) > 0
    if isinstance(v, Point):
        return v.confidence > 0.0
    return True


def _num(v: Any, line: int) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, Point):
        return v.confidence
    raise RuntimeErr(f"expected a number, found {tag(v)}", line)


def _text(v: Any) -> str:
    if v is None:
        return "none"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:g}"
    if isinstance(v, str):
        return v
    if isinstance(v, Point):
        return f'point({v.name}, conf={v.confidence:g})'
    if isinstance(v, Motion):
        return f"motion({v.name})"
    if isinstance(v, Closure):
        return f"closure({v.name})"
    if callable(v):
        return f'builtin({getattr(v, "__name__", "builtin")})'
    if isinstance(v, list):
        return "[" + ", ".join(_text(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_text(x)}" for k, x in v.items()) + "}"
    return str(v)
