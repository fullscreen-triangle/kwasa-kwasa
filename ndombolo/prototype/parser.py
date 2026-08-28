"""
Turbulance parser for the ndombolo deterministic core.

Statements are dispatched on a leading keyword; expressions use a precedence
cascade with a Pratt-style binary loop. The AST is plain dicts so it serialises
to JSON without a codec.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from lexer import LOOSE, QUANTS, Token, tokenize

Node = Dict[str, Any]

# Lowest binding first. `|>` pipes are handled at the top of the cascade.
BINARY_LEVELS: List[List[str]] = [
    ["or", "||"],
    ["and", "&&"],
    ["==", "!="],
    ["<", "<=", ">", ">="],
    ["+", "-"],
    ["*", "/", "%"],
]


class ParseError(Exception):
    def __init__(self, message: str, line: int):
        super().__init__(f"line {line}: {message}")
        self.message = message
        self.line = line


class Parser:
    def __init__(self, toks: List[Token]):
        self.toks = toks
        self.i = 0

    # -- token plumbing ---------------------------------------------------

    def peek(self, ahead: int = 0) -> Token:
        j = min(self.i + ahead, len(self.toks) - 1)
        return self.toks[j]

    def next(self) -> Token:
        t = self.toks[self.i]
        if t.kind != "eof":
            self.i += 1
        return t

    def at(self, kind: str, value: Optional[str] = None) -> bool:
        t = self.peek()
        return t.kind == kind and (value is None or t.value == value)

    def accept(self, kind: str, value: Optional[str] = None) -> Optional[Token]:
        return self.next() if self.at(kind, value) else None

    def expect(self, kind: str, value: Optional[str] = None) -> Token:
        t = self.peek()
        if not self.at(kind, value):
            want = value or kind
            raise ParseError(f"expected {want!r}, found {t.value or t.kind!r}", t.line)
        return self.next()

    def skip_newlines(self) -> None:
        while self.at("newline"):
            self.next()

    # -- program ----------------------------------------------------------

    def parse_program(self) -> List[Node]:
        body: List[Node] = []
        self.skip_newlines()
        while not self.at("eof"):
            body.append(self.parse_statement())
            self.skip_newlines()
        return body

    def parse_block(self) -> List[Node]:
        """A `:`-introduced indented block, or a single inline statement."""
        self.expect("op", ":")
        if self.accept("newline"):
            self.skip_newlines()
            self.expect("indent")
            body: List[Node] = []
            self.skip_newlines()
            while not self.at("dedent") and not self.at("eof"):
                body.append(self.parse_statement())
                self.skip_newlines()
            self.accept("dedent")
            return body
        return [self.parse_statement()]

    # -- statements -------------------------------------------------------

    def parse_statement(self) -> Node:
        t = self.peek()
        if t.kind == "kw":
            if t.value in LOOSE:
                return self.parse_loose()
            handler = getattr(self, f"parse_{t.value}", None)
            if handler is not None:
                return handler()
        return self.parse_expr_statement()

    def parse_funxn(self) -> Node:
        line = self.next().line
        name = self.expect("ident").value
        self.expect("op", "(")
        params: List[str] = []
        while not self.at("op", ")"):
            params.append(self.expect("ident").value)
            if not self.accept("op", ","):
                break
        self.expect("op", ")")
        return {"kind": "Funxn", "name": name, "params": params,
                "body": self.parse_block(), "line": line}

    def parse_item(self) -> Node:
        line = self.next().line
        name = self.expect("ident").value
        self.expect("op", "=")
        return {"kind": "Item", "name": name, "expr": self.parse_expr(),
                "line": line}

    def parse_proposition(self) -> Node:
        line = self.next().line
        name = self.expect("ident").value
        body = self.parse_block()
        motions = [s for s in body if s["kind"] == "Motion"]
        rest = [s for s in body if s["kind"] != "Motion"]
        return {"kind": "Proposition", "name": name, "motions": motions,
                "body": rest, "line": line}

    def parse_hypothesis(self) -> Node:
        line = self.next().line
        name = self.expect("ident").value
        return {"kind": "Hypothesis", "name": name, "body": self.parse_block(),
                "line": line}

    def parse_motion(self) -> Node:
        line = self.next().line
        name = self.expect("ident").value
        text = ""
        if self.accept("op", "("):
            if self.at("str"):
                text = self.next().value
            self.expect("op", ")")
        return {"kind": "Motion", "name": name, "text": text, "line": line}

    def parse_support(self) -> Node:
        return self._parse_verdict("Support")

    def parse_contradict(self) -> Node:
        return self._parse_verdict("Contradict")

    def _parse_verdict(self, kind: str) -> Node:
        line = self.next().line
        name = self.expect("ident").value
        conf: Optional[Node] = None
        if self.at("kw", "with_confidence"):
            self.next()
            self.expect("op", "(")
            conf = self.parse_expr()
            self.expect("op", ")")
        return {"kind": kind, "motion": name, "conf": conf, "line": line}

    def parse_inconclusive(self) -> Node:
        line = self.next().line
        name = self.expect("ident").value
        return {"kind": "Inconclusive", "motion": name, "line": line}

    def parse_given(self) -> Node:
        line = self.next().line
        cond = self.parse_expr()
        then = self.parse_block()
        otherwise: List[Node] = []
        save = self.i
        self.skip_newlines()
        if self.at("kw", "otherwise"):
            self.next()
            otherwise = self.parse_block()
        else:
            self.i = save
        return {"kind": "Given", "cond": cond, "then": then,
                "otherwise": otherwise, "line": line}

    def parse_within(self) -> Node:
        line = self.next().line
        target = self.parse_expr()
        return {"kind": "Within", "target": target, "body": self.parse_block(),
                "line": line}

    def parse_considering(self) -> Node:
        line = self.next().line
        quant = "item"
        if self.peek().kind in ("ident", "kw") and self.peek().value in QUANTS:
            quant = self.next().value
        elif self.at("kw", "item"):
            self.next()
        var = self.expect("ident").value
        self.expect("kw", "in")
        return {"kind": "Considering", "quant": quant, "var": var,
                "iter": self.parse_expr(), "body": self.parse_block(),
                "line": line}

    def parse_for(self) -> Node:
        line = self.next().line
        self.expect("kw", "each")
        var = self.expect("ident").value
        self.expect("kw", "in")
        return {"kind": "ForEach", "var": var, "iter": self.parse_expr(),
                "body": self.parse_block(), "line": line}

    def parse_while(self) -> Node:
        line = self.next().line
        cond = self.parse_expr()
        return {"kind": "While", "cond": cond, "body": self.parse_block(),
                "line": line}

    def parse_return(self) -> Node:
        line = self.next().line
        expr = None if self.at("newline") else self.parse_expr()
        return {"kind": "Return", "expr": expr, "line": line}

    def parse_ensure(self) -> Node:
        line = self.next().line
        return {"kind": "Ensure", "expr": self.parse_expr(), "line": line}

    def parse_point(self) -> Node:
        line = self.next().line
        name = self.expect("ident").value
        self.expect("op", "=")
        return {"kind": "Point", "name": name, "expr": self.parse_expr(),
                "line": line}

    def parse_resolve(self) -> Node:
        line = self.next().line
        return {"kind": "Resolve", "expr": self.parse_expr(), "line": line}

    def parse_import(self) -> Node:
        line = self.next().line
        parts: List[str] = []
        while not self.at("newline") and not self.at("eof"):
            parts.append(self.next().value)
        return {"kind": "Noop", "text": " ".join(parts), "line": line}

    def parse_loose(self) -> Node:
        """A keyword recognised by the grammar with no deterministic meaning."""
        t = self.next()
        parts: List[str] = [t.value]
        depth = 0
        while not self.at("eof"):
            if depth == 0 and (self.at("newline") or self.at("op", ":")):
                break
            tok = self.next()
            depth += tok.is_op("(", "[", "{") - tok.is_op(")", "]", "}")
            parts.append(tok.value)
        body = self.parse_block() if self.at("op", ":") else []
        return {"kind": "Noop", "text": " ".join(parts), "body": body,
                "line": t.line}

    def parse_expr_statement(self) -> Node:
        line = self.peek().line
        # Assignment to a bare name, distinguished by one token of lookahead.
        if self.peek().kind == "ident" and self.peek(1).is_op("="):
            name = self.next().value
            self.next()
            return {"kind": "Assign", "name": name, "expr": self.parse_expr(),
                    "line": line}
        return {"kind": "ExprStmt", "expr": self.parse_expr(), "line": line}

    # -- expressions ------------------------------------------------------

    def parse_expr(self) -> Node:
        return self.parse_pipe()

    def parse_pipe(self) -> Node:
        left = self.parse_binary(0)
        while self.at("op", "|>"):
            line = self.next().line
            right = self.parse_binary(0)
            left = {"kind": "Pipe", "left": left, "right": right, "line": line}
        return left

    def parse_binary(self, level: int) -> Node:
        if level >= len(BINARY_LEVELS):
            return self.parse_unary()
        ops = BINARY_LEVELS[level]
        left = self.parse_binary(level + 1)
        while True:
            t = self.peek()
            matched = (t.kind == "op" and t.value in ops) or \
                      (t.kind == "kw" and t.value in ops)
            if not matched:
                return left
            self.next()
            right = self.parse_binary(level + 1)
            left = {"kind": "Binary", "op": _canon(t.value), "left": left,
                    "right": right, "line": t.line}

    def parse_unary(self) -> Node:
        t = self.peek()
        if t.is_op("-") or t.is_kw("not"):
            self.next()
            return {"kind": "Unary", "op": _canon(t.value),
                    "operand": self.parse_unary(), "line": t.line}
        return self.parse_postfix()

    def parse_postfix(self) -> Node:
        node = self.parse_primary()
        while True:
            if self.at("op", "("):
                line = self.next().line
                args: List[Node] = []
                while not self.at("op", ")"):
                    args.append(self.parse_expr())
                    if not self.accept("op", ","):
                        break
                self.expect("op", ")")
                node = {"kind": "Call", "callee": node, "args": args,
                        "line": line}
            elif self.at("op", "."):
                line = self.next().line
                field = self.expect("ident").value
                node = {"kind": "Field", "target": node, "field": field,
                        "line": line}
            elif self.at("op", "["):
                line = self.next().line
                index = self.parse_expr()
                self.expect("op", "]")
                node = {"kind": "Index", "target": node, "index": index,
                        "line": line}
            else:
                return node

    def parse_primary(self) -> Node:
        t = self.next()

        if t.kind == "num":
            return {"kind": "Num", "value": float(t.value), "line": t.line}
        if t.kind == "str":
            return {"kind": "Str", "value": t.value, "line": t.line}
        if t.kind == "ident":
            if t.value in ("true", "false"):
                return {"kind": "Bool", "value": t.value == "true",
                        "line": t.line}
            if t.value == "none":
                return {"kind": "None", "line": t.line}
            return {"kind": "Var", "name": t.value, "line": t.line}

        if t.is_op("("):
            inner = self.parse_expr()
            self.expect("op", ")")
            return inner

        if t.is_op("["):
            items: List[Node] = []
            self.skip_newlines()
            while not self.at("op", "]"):
                items.append(self.parse_expr())
                self.skip_newlines()
                if not self.accept("op", ","):
                    break
                self.skip_newlines()
            self.expect("op", "]")
            return {"kind": "List", "items": items, "line": t.line}

        if t.is_op("{"):
            entries: List[Dict[str, Any]] = []
            self.skip_newlines()
            while not self.at("op", "}"):
                key_tok = self.next()
                if key_tok.kind not in ("ident", "str", "kw"):
                    raise ParseError("expected map key", key_tok.line)
                self.expect("op", ":")
                entries.append({"key": key_tok.value, "value": self.parse_expr()})
                self.skip_newlines()
                if not self.accept("op", ","):
                    break
                self.skip_newlines()
            self.expect("op", "}")
            return {"kind": "Map", "entries": entries, "line": t.line}

        raise ParseError(f"unexpected {t.value or t.kind!r}", t.line)


def _canon(op: str) -> str:
    return {"&&": "and", "||": "or"}.get(op, op)


def parse(source: str, first_line: int = 1) -> List[Node]:
    """Parse `source`; `first_line` makes sites script-absolute across cells."""
    return Parser(tokenize(source, first_line)).parse_program()
