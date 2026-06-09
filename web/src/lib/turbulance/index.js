/* ==========================================================================
 *  Turbulance — a deterministic-core interpreter for the kwasa-kwasa playground
 *
 *  Implements the deterministic fragment of the language specified in
 *  "Turbulance: A Formal Specification" (domain-grammar-specification):
 *    - funxn / item / given(/otherwise) / within / considering / for each / while
 *    - proposition / motion / support / contradict   (argumentation semantics)
 *    - point / resolve                               (resolution semantics)
 *    - the noisy-or confidence algebra
 *    - expressions with the full precedence cascade and pipe operators
 *
 *  No backend, no LLM: world-touching resolvers are out of scope for v1.
 *  Entry point:  run(source) -> { ok, output, propositions, points, value, error }
 * ========================================================================== */

/* ----------------------------- 1. LEXER ----------------------------------- */

const KEYWORDS = new Set([
  "funxn", "item", "proposition", "hypothesis", "motion", "support", "contradict",
  "inconclusive", "within", "given", "considering", "for", "each", "in",
  "while", "return", "ensure", "allow", "research", "cause", "point",
  "resolution", "resolve", "cycle", "drift", "flow", "roll", "until",
  "settled", "over", "on", "goal", "metacognitive", "import", "from", "as",
  "otherwise", "with_confidence", "and", "or", "not",
]);

const QUANTS = new Set(["all", "these"]); // considering quantifiers (item is default)

// Count net bracket depth of a text fragment, ignoring brackets inside strings.
function bracketDelta(text) {
  let depth = 0, inStr = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inStr) {
      if (c === "\\") { i++; continue; }
      if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') { inStr = true; continue; }
    if (c === "/" && text[i + 1] === "/") break;
    if (c === "#") break;
    if (c === "(" || c === "[" || c === "{") depth++;
    else if (c === ")" || c === "]" || c === "}") depth--;
  }
  return depth;
}

function leadingIndent(line) {
  let n = 0;
  for (const c of line) {
    if (c === " ") n++;
    else if (c === "\t") n += 4;
    else break;
  }
  return n;
}

// Tokenize one logical line's content into the token list.
function tokenizeLine(text, line, out) {
  let i = 0;
  const n = text.length;
  const push = (type, value) => out.push({ type, value, line });
  while (i < n) {
    const c = text[i];
    if (c === " " || c === "\t" || c === "\n" || c === "\r") { i++; continue; }
    if (c === "/" && text[i + 1] === "/") break;
    if (c === "#") break;
    // string
    if (c === '"') {
      let s = "", j = i + 1;
      while (j < n && text[j] !== '"') {
        if (text[j] === "\\") {
          const e = text[j + 1];
          s += e === "n" ? "\n" : e === "t" ? "\t" : e === "r" ? "\r" : e;
          j += 2;
        } else { s += text[j]; j++; }
      }
      push("STR", s);
      i = j + 1;
      continue;
    }
    // number
    if (c >= "0" && c <= "9") {
      let j = i;
      while (j < n && text[j] >= "0" && text[j] <= "9") j++;
      if (text[j] === "." && text[j + 1] >= "0" && text[j + 1] <= "9") {
        j++;
        while (j < n && text[j] >= "0" && text[j] <= "9") j++;
      }
      push("NUM", parseFloat(text.slice(i, j)));
      i = j;
      continue;
    }
    // identifier / keyword
    if (/[A-Za-z_]/.test(c)) {
      let j = i;
      while (j < n && /[A-Za-z0-9_]/.test(text[j])) j++;
      const word = text.slice(i, j);
      if (word === "true") push("BOOL", true);
      else if (word === "false") push("BOOL", false);
      else if (word === "none") push("NONE", null);
      else if (KEYWORDS.has(word)) push("KW", word);
      else push("ID", word);
      i = j;
      continue;
    }
    // operators (multi-char first)
    const two = text.slice(i, i + 2);
    if (["==", "!=", "<=", ">=", "&&", "||", "|>", "=>"].includes(two)) {
      push("OP", two); i += 2; continue;
    }
    if ("+-*/<>=|!.,:;(){}[]".includes(c)) { push("OP", c); i++; continue; }
    throw new TbError(`Unexpected character '${c}'`, line);
  }
}

function lex(src) {
  const physical = src.replace(/\r\n?/g, "\n").split("\n");
  // Build logical lines (joining bracket continuations) with indentation.
  const logical = [];
  for (let i = 0; i < physical.length; i++) {
    const raw = physical[i];
    const indent = leadingIndent(raw);
    const content = raw.slice(raw.length - raw.trimStart().length);
    const trimmed = content.trim();
    if (trimmed === "" || trimmed.startsWith("//") || trimmed.startsWith("#")) continue;
    let text = content;
    let depth = bracketDelta(content);
    while (depth > 0 && i + 1 < physical.length) {
      i++;
      text += "\n" + physical[i];
      depth += bracketDelta(physical[i]);
    }
    logical.push({ indent, text, line: i + 1 });
  }
  // Emit tokens with NEWLINE / INDENT / DEDENT.
  const tokens = [];
  const stack = [0];
  for (const L of logical) {
    if (L.indent > stack[stack.length - 1]) {
      stack.push(L.indent);
      tokens.push({ type: "INDENT", value: null, line: L.line });
    } else {
      while (L.indent < stack[stack.length - 1]) {
        stack.pop();
        tokens.push({ type: "DEDENT", value: null, line: L.line });
      }
    }
    tokenizeLine(L.text, L.line, tokens);
    tokens.push({ type: "NEWLINE", value: null, line: L.line });
  }
  while (stack.length > 1) { stack.pop(); tokens.push({ type: "DEDENT", value: null, line: 0 }); }
  tokens.push({ type: "EOF", value: null, line: 0 });
  return tokens;
}

/* ----------------------------- 2. PARSER ---------------------------------- */

class TbError extends Error {
  constructor(message, line) { super(message); this.line = line; }
}

class Parser {
  constructor(tokens) { this.toks = tokens; this.pos = 0; }
  peek(k = 0) { return this.toks[this.pos + k]; }
  next() { return this.toks[this.pos++]; }
  at(type, value) {
    const t = this.peek();
    return t.type === type && (value === undefined || t.value === value);
  }
  atKW(w) { return this.at("KW", w); }
  atOP(o) { return this.at("OP", o); }
  eat(type, value) {
    if (!this.at(type, value)) {
      const t = this.peek();
      throw new TbError(`Expected ${value ?? type}, got ${t.value ?? t.type}`, t.line);
    }
    return this.next();
  }
  skipNewlines() { while (this.at("NEWLINE")) this.next(); }

  parseProgram() {
    const body = [];
    this.skipNewlines();
    while (!this.at("EOF")) {
      body.push(this.parseStmt());
      this.skipNewlines();
    }
    return { kind: "Program", body };
  }

  // A block is: NEWLINE INDENT stmt* DEDENT  (the colon has already been eaten)
  parseBlock() {
    this.skipNewlines();
    this.eat("INDENT");
    const body = [];
    this.skipNewlines();
    while (!this.at("DEDENT") && !this.at("EOF")) {
      body.push(this.parseStmt());
      this.skipNewlines();
    }
    if (this.at("DEDENT")) this.next();
    return { kind: "Block", body };
  }

  parseStmt() {
    const t = this.peek();
    if (t.type === "KW") {
      switch (t.value) {
        case "funxn": return this.parseFunxn();
        case "proposition": return this.parseProposition();
        case "hypothesis": return this.parseHypothesis();
        case "motion": return this.parseMotion();
        case "item": return this.parseItem();
        case "given": return this.parseGiven();
        case "within": return this.parseWithin();
        case "considering": return this.parseConsidering();
        case "for": return this.parseFor();
        case "while": return this.parseWhile();
        case "return": return this.parseReturn();
        case "ensure": { this.next(); return { kind: "Ensure", expr: this.parseExpr(), line: t.line }; }
        case "support": return this.parseSupport(false);
        case "contradict": return this.parseSupport(true);
        case "inconclusive": { this.next(); this.eat("OP", "("); const m = this.eat("STR").value; this.eat("OP", ")"); return { kind: "Inconclusive", message: m }; }
        case "point": return this.parsePoint();
        case "import": return this.parseImport();
        case "allow": case "research": case "cause": case "goal":
        case "metacognitive": case "resolution":
          // Recognised but treated as no-ops / expression-ish in v1.
          return this.parseLooseKeywordStmt();
        default: break;
      }
    }
    // assignment or expression statement
    const expr = this.parseExpr();
    if (this.atOP("=")) {
      this.next();
      const value = this.parseExpr();
      return { kind: "Assign", target: expr, value, line: t.line };
    }
    return { kind: "ExprStmt", expr, line: t.line };
  }

  parseFunxn() {
    const line = this.next().line; // funxn
    const name = this.eat("ID").value;
    this.eat("OP", "(");
    const params = [];
    if (!this.atOP(")")) {
      do {
        const pname = this.eat("ID").value;
        let ptype = null, def = null;
        if (this.atOP(":")) { this.next(); ptype = this.parseTypeName(); }
        if (this.atOP("=")) { this.next(); def = this.parseExpr(); }
        params.push({ name: pname, ptype, def });
      } while (this.atOP(",") && this.next());
    }
    this.eat("OP", ")");
    this.eat("OP", ":");
    const body = this.parseBlock();
    return { kind: "Funxn", name, params, body, line };
  }

  parseTypeName() {
    // consume a (possibly generic) type name; we ignore types in v1
    let t = this.next().value;
    if (this.atOP("[")) { this.next(); this.parseTypeName(); this.eat("OP", "]"); }
    return t;
  }

  parseProposition() {
    const line = this.next().line; // proposition
    const name = this.eat("ID").value;
    this.eat("OP", ":");
    const block = this.parseBlock();
    const motions = block.body.filter((s) => s.kind === "Motion");
    const body = block.body.filter((s) => s.kind !== "Motion");
    return { kind: "Proposition", name, motions, body, line };
  }

  // hypothesis Name:
  //     claim: "..."
  //     success_criteria:
  //         - sensitivity: 0.85
  parseHypothesis() {
    const line = this.next().line; // hypothesis
    const name = this.eat("ID").value;
    this.eat("OP", ":");
    this.skipNewlines();
    this.eat("INDENT");
    const fields = [];
    this.skipNewlines();
    while (!this.at("DEDENT") && !this.at("EOF")) {
      const key = this.eat("ID").value;
      this.eat("OP", ":");
      if (this.at("NEWLINE")) {
        // nested block of "- subkey: value" entries -> a map
        this.skipNewlines();
        const sub = [];
        if (this.at("INDENT")) {
          this.next();
          this.skipNewlines();
          while (!this.at("DEDENT") && !this.at("EOF")) {
            if (this.atOP("-")) this.next();
            const subkey = this.eat("ID").value;
            this.eat("OP", ":");
            sub.push({ key: subkey, value: this.parseExpr() });
            this.skipNewlines();
          }
          if (this.at("DEDENT")) this.next();
        }
        fields.push({ key, nested: sub });
      } else {
        fields.push({ key, value: this.parseExpr() });
        this.skipNewlines();
      }
    }
    if (this.at("DEDENT")) this.next();
    return { kind: "Hypothesis", name, fields, line };
  }

  parseMotion() {
    const line = this.next().line; // motion
    const name = this.eat("ID").value;
    this.eat("OP", "(");
    const desc = this.eat("STR").value;
    this.eat("OP", ")");
    return { kind: "Motion", name, desc, line };
  }

  parseSupport(isContradict) {
    const line = this.next().line; // support/contradict
    const motion = this.eat("ID").value;
    let conf = null;
    if (this.atKW("with_confidence")) {
      this.next(); this.eat("OP", "("); conf = this.parseExpr(); this.eat("OP", ")");
    }
    return { kind: isContradict ? "Contradict" : "Support", motion, conf, line };
  }

  parseItem() {
    const line = this.next().line; // item
    const name = this.eat("ID").value;
    if (this.atOP(":")) { this.next(); this.parseTypeName(); }
    this.eat("OP", "=");
    const value = this.parseExpr();
    return { kind: "Item", name, value, line };
  }

  parseGiven() {
    const line = this.next().line; // given
    const cond = this.parseExpr();
    this.eat("OP", ":");
    const then = this.parseBlock();
    let els = null;
    // optional: (newlines) given otherwise : block
    const save = this.pos;
    this.skipNewlines();
    if (this.atKW("given") && this.peek(1).type === "KW" && this.peek(1).value === "otherwise") {
      this.next(); this.next(); this.eat("OP", ":");
      els = this.parseBlock();
    } else {
      this.pos = save;
    }
    return { kind: "Given", cond, then, els, line };
  }

  parseWithin() {
    const line = this.next().line; // within
    const target = this.parseExpr();
    let alias = null;
    if (this.atKW("as")) { this.next(); alias = this.eat("ID").value; }
    this.eat("OP", ":");
    const body = this.parseBlock();
    return { kind: "Within", target, alias, body, line };
  }

  parseConsidering() {
    const line = this.next().line; // considering
    let quant = "item";
    if (this.at("ID") && QUANTS.has(this.peek().value)) quant = this.next().value;
    const variable = this.eat("ID").value;
    this.eat("KW", "in");
    const iterable = this.parseExpr();
    this.eat("OP", ":");
    const body = this.parseBlock();
    return { kind: "Considering", quant, variable, iterable, body, line };
  }

  parseFor() {
    const line = this.next().line; // for
    this.eat("KW", "each");
    const variable = this.eat("ID").value;
    this.eat("KW", "in");
    const iterable = this.parseExpr();
    this.eat("OP", ":");
    const body = this.parseBlock();
    return { kind: "ForEach", variable, iterable, body, line };
  }

  parseWhile() {
    const line = this.next().line; // while
    const cond = this.parseExpr();
    this.eat("OP", ":");
    const body = this.parseBlock();
    return { kind: "While", cond, body, line };
  }

  parseReturn() {
    const line = this.next().line; // return
    let expr = null;
    if (!this.at("NEWLINE") && !this.at("DEDENT") && !this.at("EOF")) expr = this.parseExpr();
    return { kind: "Return", expr, line };
  }

  parsePoint() {
    const line = this.next().line; // point
    const name = this.eat("ID").value;
    this.eat("OP", "=");
    const value = this.parsePrimary(); // a map literal
    return { kind: "Item", name, value, line, isPoint: true };
  }

  parseImport() {
    // consume the rest of the logical line; imports are no-ops in v1
    const line = this.next().line;
    while (!this.at("NEWLINE") && !this.at("EOF")) this.next();
    return { kind: "Noop", line };
  }

  parseLooseKeywordStmt() {
    const line = this.next().line;
    // consume to end of line as a no-op (recognised but unimplemented in v1)
    while (!this.at("NEWLINE") && !this.at("INDENT") && !this.at("EOF")) this.next();
    if (this.at("INDENT")) this.parseBlock();
    return { kind: "Noop", line };
  }

  /* ---- expressions (precedence cascade) ---- */
  parseExpr() { return this.parsePipe(); }

  parsePipe() {
    let left = this.parseOr();
    while (this.atOP("|") || this.atOP("|>")) {
      const line = this.next().line;
      const right = this.parseOr();
      // a | f       -> f(a)
      // a | g(b,c)  -> g(a,b,c)
      if (right.kind === "Call") {
        left = { kind: "Call", callee: right.callee, args: [left, ...right.args], line };
      } else {
        left = { kind: "Call", callee: right, args: [left], line };
      }
    }
    return left;
  }
  parseOr() {
    let left = this.parseAnd();
    while (this.atOP("||") || this.atKW("or")) { const line = this.next().line; left = { kind: "Logic", op: "||", left, right: this.parseAnd(), line }; }
    return left;
  }
  parseAnd() {
    let left = this.parseEquality();
    while (this.atOP("&&") || this.atKW("and")) { const line = this.next().line; left = { kind: "Logic", op: "&&", left, right: this.parseEquality(), line }; }
    return left;
  }
  parseEquality() {
    let left = this.parseComparison();
    while (this.atOP("==") || this.atOP("!=")) { const op = this.next().value; left = { kind: "Binary", op, left, right: this.parseComparison() }; }
    return left;
  }
  parseComparison() {
    let left = this.parseTerm();
    while (this.atOP("<") || this.atOP(">") || this.atOP("<=") || this.atOP(">=")) { const op = this.next().value; left = { kind: "Binary", op, left, right: this.parseTerm() }; }
    return left;
  }
  parseTerm() {
    let left = this.parseFactor();
    while (this.atOP("+") || this.atOP("-")) { const op = this.next().value; left = { kind: "Binary", op, left, right: this.parseFactor() }; }
    return left;
  }
  parseFactor() {
    let left = this.parseUnary();
    while (this.atOP("*") || this.atOP("/")) { const op = this.next().value; left = { kind: "Binary", op, left, right: this.parseUnary() }; }
    return left;
  }
  parseUnary() {
    if (this.atOP("!") || this.atKW("not")) { const line = this.next().line; return { kind: "Unary", op: "!", operand: this.parseUnary(), line }; }
    if (this.atOP("-")) { const line = this.next().line; return { kind: "Unary", op: "-", operand: this.parseUnary(), line }; }
    if (this.atKW("resolve")) { const line = this.next().line; return { kind: "Resolve", expr: this.parseUnary(), line }; }
    return this.parseCall();
  }
  parseCall() {
    let node = this.parsePrimary();
    while (true) {
      if (this.atOP("(")) {
        this.next();
        const args = [];
        if (!this.atOP(")")) { do { args.push(this.parseExpr()); } while (this.atOP(",") && this.next()); }
        this.eat("OP", ")");
        node = { kind: "Call", callee: node, args };
      } else if (this.atOP(".")) {
        this.next();
        const prop = this.eat("ID").value;
        node = { kind: "Member", object: node, prop };
      } else if (this.atOP("[")) {
        this.next();
        const index = this.parseExpr();
        this.eat("OP", "]");
        node = { kind: "Index", object: node, index };
      } else break;
    }
    return node;
  }
  parsePrimary() {
    const t = this.peek();
    if (t.type === "NUM") { this.next(); return { kind: "Num", value: t.value }; }
    if (t.type === "STR") { this.next(); return { kind: "Str", value: t.value }; }
    if (t.type === "BOOL") { this.next(); return { kind: "Bool", value: t.value }; }
    if (t.type === "NONE") { this.next(); return { kind: "None" }; }
    if (t.type === "ID") { this.next(); return { kind: "Var", name: t.value }; }
    if (this.atOP("(")) { this.next(); const e = this.parseExpr(); this.eat("OP", ")"); return e; }
    if (this.atOP("[")) {
      this.next();
      const elements = [];
      if (!this.atOP("]")) { do { elements.push(this.parseExpr()); } while (this.atOP(",") && this.next()); }
      this.eat("OP", "]");
      return { kind: "List", elements };
    }
    if (this.atOP("{")) {
      this.next();
      const fields = [];
      if (!this.atOP("}")) {
        do {
          const key = this.at("STR") ? this.next().value : this.eat("ID").value;
          this.eat("OP", ":");
          fields.push({ key, value: this.parseExpr() });
        } while (this.atOP(",") && this.next());
      }
      this.eat("OP", "}");
      return { kind: "Map", fields };
    }
    throw new TbError(`Unexpected token '${t.value ?? t.type}'`, t.line);
  }
}

/* --------------------------- 3. VALUES ------------------------------------ */

const noisyOr = (c, d) => 1 - (1 - c) * (1 - d);
const clip01 = (x) => Math.max(0, Math.min(1, x));

function tbType(v) {
  if (v === null || v === undefined) return "none";
  if (typeof v === "number") return "num";
  if (typeof v === "string") return "str";
  if (typeof v === "boolean") return "bool";
  if (Array.isArray(v)) return "list";
  if (v.__t) return v.__t;
  return "map";
}

function isTruthy(v) {
  const t = tbType(v);
  if (t === "bool") return v;
  if (t === "none") return false;
  if (t === "num") return v !== 0;
  if (t === "str") return v.length > 0;
  if (t === "list") return v.length > 0;
  return true;
}

function tbToString(v) {
  const t = tbType(v);
  switch (t) {
    case "none": return "none";
    case "num": return Number.isInteger(v) ? String(v) : String(v);
    case "str": return v;
    case "bool": return v ? "true" : "false";
    case "list": return "[" + v.map(tbToString).join(", ") + "]";
    case "map": return "{" + Object.entries(v.data).map(([k, x]) => `${k}: ${tbToString(x)}`).join(", ") + "}";
    case "point": return `point("${v.content}", conf=${v.confidence.toFixed(2)})`;
    case "entity": return `entity(${tbToString(v.value)}, conf=${v.confidence.toFixed(2)})`;
    case "closure": return `funxn ${v.name || "<anon>"}`;
    default: return String(v);
  }
}

/* --------------------------- 4. INTERPRETER ------------------------------- */

class Env {
  constructor(parent = null) { this.vars = new Map(); this.parent = parent; }
  get(name) {
    if (this.vars.has(name)) return this.vars.get(name);
    if (this.parent) return this.parent.get(name);
    throw new TbError(`Undefined variable '${name}'`);
  }
  has(name) { return this.vars.has(name) || (this.parent ? this.parent.has(name) : false); }
  define(name, v) { this.vars.set(name, v); }
  set(name, v) {
    let e = this;
    while (e) { if (e.vars.has(name)) { e.vars.set(name, v); return; } e = e.parent; }
    this.vars.set(name, v); // implicit declaration
  }
}

class ReturnSignal { constructor(value) { this.value = value; } }

class Interpreter {
  constructor(opts = {}) {
    this.output = [];
    this.propositions = [];
    this.points = [];
    this.debateStack = [];
    this.global = new Env();
    this.thresholds = { plus: 0.7, minus: 0.3 };
    this.files = opts.files || {};                 // filename -> source (for delegation)
    this.onStatus = opts.onStatus || (() => {});   // progress callback (e.g. loading Python)
    // polyglot/AI namespaces (Paper B tool/oracle resolvers)
    this.global.define("trebuchet", {
      __t: "map",
      data: { delegate: { __t: "builtin", name: "trebuchet.delegate" } },
    });
  }

  async run(program) {
    for (const stmt of program.body) {
      // hoist function declarations so order doesn't matter
      if (stmt.kind === "Funxn") await this.execStmt(stmt, this.global);
    }
    for (const stmt of program.body) {
      if (stmt.kind !== "Funxn") await this.execStmt(stmt, this.global);
    }
    // If a main() exists, call it.
    if (this.global.has("main")) {
      const m = this.global.get("main");
      if (tbType(m) === "closure") await this.callClosure(m, []);
    }
  }

  async execBlock(block, env) {
    for (const s of block.body) await this.execStmt(s, env);
  }

  async execStmt(node, env) {
    switch (node.kind) {
      case "Funxn":
        env.define(node.name, { __t: "closure", name: node.name, params: node.params, body: node.body, env });
        return;
      case "Item": {
        let v = await this.eval(node.value, env);
        if (node.isPoint) v = this.makePoint(v);
        env.define(node.name, v);
        return;
      }
      case "Hypothesis": {
        const data = {};
        for (const f of node.fields) {
          if (f.nested) {
            const m = {};
            for (const s of f.nested) m[s.key] = await this.eval(s.value, env);
            data[f.key] = { __t: "map", data: m };
          } else {
            data[f.key] = await this.eval(f.value, env);
          }
        }
        env.define(node.name, { __t: "map", data });
        return;
      }
      case "Assign": {
        const v = await this.eval(node.value, env);
        await this.assign(node.target, v, env);
        return;
      }
      case "ExprStmt":
        await this.eval(node.expr, env);
        return;
      case "Return":
        throw new ReturnSignal(node.expr ? await this.eval(node.expr, env) : null);
      case "Ensure":
        if (!isTruthy(await this.eval(node.expr, env))) throw new TbError("ensure failed: assertion did not hold", node.line);
        return;
      case "Given": {
        if (isTruthy(await this.eval(node.cond, env))) await this.execBlock(node.then, new Env(env));
        else if (node.els) await this.execBlock(node.els, new Env(env));
        return;
      }
      case "Within": {
        const target = await this.eval(node.target, env);
        const inner = new Env(env);
        if (node.alias) inner.define(node.alias, target);
        await this.execBlock(node.body, inner);
        return;
      }
      case "ForEach": {
        const it = this.iterableOf(await this.eval(node.iterable, env));
        for (const x of it) { const inner = new Env(env); inner.define(node.variable, x); await this.execBlock(node.body, inner); }
        return;
      }
      case "Considering": {
        const it = this.iterableOf(await this.eval(node.iterable, env));
        for (const x of it) { const inner = new Env(env); inner.define(node.variable, x); await this.execBlock(node.body, inner); }
        return;
      }
      case "While": {
        let guard = 0;
        while (isTruthy(await this.eval(node.cond, env))) {
          await this.execBlock(node.body, new Env(env));
          if (++guard > 100000) throw new TbError("while loop exceeded 100000 iterations", node.line);
        }
        return;
      }
      case "Proposition": return this.execProposition(node, env);
      case "Support": case "Contradict": return this.execSupport(node, env);
      case "Inconclusive": case "Noop": return;
      default:
        throw new TbError(`Cannot execute statement '${node.kind}'`, node.line);
    }
  }

  async assign(target, value, env) {
    if (target.kind === "Var") { env.set(target.name, value); return; }
    if (target.kind === "Index") {
      const obj = await this.eval(target.object, env);
      const idx = await this.eval(target.index, env);
      if (Array.isArray(obj)) obj[idx] = value;
      else if (tbType(obj) === "map") obj.data[idx] = value;
      return;
    }
    if (target.kind === "Member") {
      const obj = await this.eval(target.object, env);
      if (tbType(obj) === "map") obj.data[target.prop] = value;
      return;
    }
    throw new TbError("Invalid assignment target", target.line);
  }

  async execProposition(node, env) {
    const debate = {};
    for (const m of node.motions) debate[m.name] = { desc: m.desc, aff: [], con: [] };
    this.debateStack.push(debate);
    const inner = new Env(env);
    try { for (const s of node.body) await this.execStmt(s, inner); }
    finally { this.debateStack.pop(); }

    const motions = node.motions.map((m) => {
      const d = debate[m.name];
      const score = this.scoreMotion(d);
      const verdict = score >= this.thresholds.plus ? "Supported"
        : score <= this.thresholds.minus ? "Contradicted" : "Inconclusive";
      return { name: m.name, desc: m.desc, score, verdict };
    });
    const overall = motions.length ? motions.reduce((acc, m) => noisyOr(acc, m.score), 0) : 0;
    const verdict = overall >= this.thresholds.plus ? "Supported"
      : overall <= this.thresholds.minus ? "Contradicted" : "Inconclusive";
    const result = { __t: "map", data: { name: node.name, verdict, score: overall } };
    env.define(node.name, result);
    this.propositions.push({ name: node.name, verdict, score: overall, motions });
    return;
  }

  // MaxLik motion score: clipped weighted (affirmation - contention) means.
  scoreMotion(d) {
    let sPlus = 0, sMinus = 0, w = 0;
    for (const a of d.aff) { sPlus += a.weight * a.strength * a.conf; w += a.weight; }
    for (const c of d.con) { sMinus += c.weight * c.strength * c.conf; w += c.weight; }
    if (w === 0) return 0;
    return clip01((sPlus - sMinus) / w);
  }

  async execSupport(node, env) {
    if (this.debateStack.length === 0) throw new TbError(`'${node.kind.toLowerCase()}' outside a proposition`, node.line);
    const debate = this.debateStack[this.debateStack.length - 1];
    const entry = debate[node.motion];
    if (!entry) throw new TbError(`Unknown motion '${node.motion}'`, node.line);
    const conf = node.conf == null ? 1 : clip01(Number(await this.eval(node.conf, env)));
    const item = { strength: conf, conf, weight: 1 };
    if (node.kind === "Support") entry.aff.push(item); else entry.con.push(item);
  }

  iterableOf(v) {
    const t = tbType(v);
    if (t === "list") return v;
    if (t === "str") return v.split(/(?<=[.!?])\s+/).filter((s) => s.trim().length); // sentences
    if (t === "map") return Object.values(v.data);
    if (t === "none") return [];
    return [v];
  }

  /* ---- expression evaluation ---- */
  async eval(node, env) {
    switch (node.kind) {
      case "Num": return node.value;
      case "Str": return node.value;
      case "Bool": return node.value;
      case "None": return null;
      case "Var": {
        if (env.has(node.name)) return env.get(node.name);
        if (BUILTINS[node.name]) return { __t: "builtin", name: node.name };
        throw new TbError(`Undefined variable '${node.name}'`, node.line);
      }
      case "List": {
        const out = [];
        for (const e of node.elements) out.push(await this.eval(e, env));
        return out;
      }
      case "Map": {
        const data = {};
        for (const f of node.fields) data[f.key] = await this.eval(f.value, env);
        return { __t: "map", data };
      }
      case "Unary": {
        const v = await this.eval(node.operand, env);
        if (node.op === "!") return !isTruthy(v);
        return -Number(v);
      }
      case "Logic": {
        const l = await this.eval(node.left, env);
        if (node.op === "&&") return isTruthy(l) ? await this.eval(node.right, env) : l;
        return isTruthy(l) ? l : await this.eval(node.right, env);
      }
      case "Binary": return this.binop(node.op, await this.eval(node.left, env), await this.eval(node.right, env), node);
      case "Member": return this.member(await this.eval(node.object, env), node.prop, node);
      case "Index": {
        const obj = await this.eval(node.object, env);
        const idx = await this.eval(node.index, env);
        if (Array.isArray(obj)) return obj[idx] ?? null;
        if (tbType(obj) === "str") return obj[idx] ?? "";
        if (tbType(obj) === "map") return obj.data[idx] ?? null;
        throw new TbError("Cannot index this value", node.line);
      }
      case "Call": return this.evalCall(node, env);
      case "Resolve": return this.resolve(await this.eval(node.expr, env), node);
      default: throw new TbError(`Cannot evaluate '${node.kind}'`, node.line);
    }
  }

  binop(op, l, r, node) {
    switch (op) {
      case "+":
        if (tbType(l) === "str" || tbType(r) === "str") return tbToString(l) + tbToString(r);
        if (Array.isArray(l) && Array.isArray(r)) return [...l, ...r];
        return Number(l) + Number(r);
      case "-": return Number(l) - Number(r);
      case "*": return Number(l) * Number(r);
      case "/": return Number(l) / Number(r);
      case "==": return this.equals(l, r);
      case "!=": return !this.equals(l, r);
      case "<": return Number(l) < Number(r);
      case ">": return Number(l) > Number(r);
      case "<=": return Number(l) <= Number(r);
      case ">=": return Number(l) >= Number(r);
      default: throw new TbError(`Unknown operator ${op}`, node && node.line);
    }
  }
  equals(l, r) {
    const tl = tbType(l), tr = tbType(r);
    if (tl !== tr) return false;
    if (tl === "num" || tl === "str" || tl === "bool" || tl === "none") return l === r;
    return l === r;
  }

  member(obj, prop, node) {
    const t = tbType(obj);
    if (t === "map") return obj.data[prop] ?? null;
    if (t === "entity") { if (prop === "confidence") return obj.confidence; if (prop === "value") return obj.value; }
    if (t === "point") { if (prop === "confidence") return obj.confidence; if (prop === "content") return obj.content; }
    if (t === "str") { if (prop === "length") return obj.length; }
    if (t === "list") { if (prop === "length") return obj.length; }
    return null;
  }

  async evalCall(node, env) {
    const callee = await this.eval(node.callee, env);
    const args = [];
    for (const a of node.args) args.push(await this.eval(a, env));
    const t = tbType(callee);
    if (t === "closure") return await this.callClosure(callee, args);
    if (t === "builtin") return await BUILTINS[callee.name](this, args, node);
    throw new TbError("Attempt to call a non-function", node.line);
  }

  async callClosure(clo, args) {
    const env = new Env(clo.env);
    for (let i = 0; i < clo.params.length; i++) {
      const p = clo.params[i];
      let v = args[i];
      if (v === undefined) v = p.def ? await this.eval(p.def, env) : null;
      env.define(p.name, v);
    }
    try { await this.execBlock(clo.body, env); }
    catch (e) { if (e instanceof ReturnSignal) return e.value; throw e; }
    return null;
  }

  // Build a point value from a map literal (or primitive) following `point x = {...}`.
  makePoint(v) {
    if (tbType(v) === "point") return v;
    if (tbType(v) === "map") {
      const d = v.data;
      const content = d.content != null ? tbToString(d.content) : "";
      const confidence = d.confidence != null ? clip01(Number(d.confidence)) : 1;
      let interps = [];
      if (Array.isArray(d.interpretations)) {
        interps = d.interpretations.map((it) => {
          const m = tbType(it) === "map" ? it.data : {};
          return { meaning: tbToString(m.meaning ?? m.content ?? ""), prob: Number(m.probability ?? m.prob ?? 0) };
        });
      }
      return { __t: "point", content, confidence, interps };
    }
    return { __t: "point", content: tbToString(v), confidence: 1, interps: [] };
  }

  // Deterministic resolution (maximum-likelihood strategy).
  resolve(v, node) {
    const t = tbType(v);
    if (t === "entity") return v;
    if (t !== "point") {
      // resolving a non-point is the identity wrapped as an entity
      return { __t: "entity", value: v, confidence: 1 };
    }
    const interps = v.interps || [];
    let value = v.content, confidence = v.confidence ?? 1;
    if (interps.length > 0) {
      const total = interps.reduce((s, i) => s + i.prob, 0) || 1;
      const norm = interps.map((i) => ({ meaning: i.meaning, p: i.prob / total }));
      const dom = norm.reduce((a, b) => (b.p > a.p ? b : a));
      // interpretation entropy -> ambiguity -> confidence
      const H = -norm.reduce((s, i) => s + (i.p > 0 ? i.p * Math.log2(i.p) : 0), 0);
      const Hmax = Math.log2(norm.length) || 1;
      const ambiguity = norm.length > 1 ? H / Hmax : 0;
      value = dom.meaning;
      confidence = clip01(v.confidence != null ? v.confidence * (1 - ambiguity) : 1 - ambiguity);
    }
    const ent = { __t: "entity", value, confidence };
    this.points.push({ content: v.content, confidence: ent.confidence, value });
    return ent;
  }
}

/* --------------------------- 5. BUILTINS ---------------------------------- */

// Convert a Turbulance value to plain JS (for passing into a Python specialist).
function tbToJS(v) {
  const t = tbType(v);
  if (t === "none") return null;
  if (t === "num" || t === "str" || t === "bool") return v;
  if (t === "list") return v.map(tbToJS);
  if (t === "map") { const o = {}; for (const k of Object.keys(v.data)) o[k] = tbToJS(v.data[k]); return o; }
  if (t === "entity") return { value: tbToJS(v.value), confidence: v.confidence };
  if (t === "point") return { content: v.content, confidence: v.confidence };
  return null;
}

// Convert a plain-JS value (a Python specialist's result) back to a Turbulance value.
function jsToTb(x) {
  if (x === null || x === undefined) return null;
  const tp = typeof x;
  if (tp === "number" || tp === "string" || tp === "boolean") return x;
  if (Array.isArray(x)) return x.map(jsToTb);
  if (x instanceof Map) { const data = {}; for (const [k, val] of x) data[String(k)] = jsToTb(val); return { __t: "map", data }; }
  if (tp === "object") { const data = {}; for (const k of Object.keys(x)) data[k] = jsToTb(x[k]); return { __t: "map", data }; }
  return null;
}

const BUILTINS = {
  print(interp, args) {
    if (args.length > 0 && tbType(args[0]) === "str" && args[0].includes("{}")) {
      let s = args[0], i = 1;
      s = s.replace(/\{\}/g, () => (i < args.length ? tbToString(args[i++]) : "{}"));
      interp.output.push(s);
    } else {
      interp.output.push(args.map(tbToString).join(" "));
    }
    return null;
  },
  len(_i, args) { const v = args[0]; const t = tbType(v); return t === "str" || t === "list" ? v.length : 0; },
  contains(_i, args) { const [s, sub] = args; return tbType(s) === "str" ? s.includes(sub) : Array.isArray(s) ? s.includes(sub) : false; },
  matches(_i, args) { const [s, pat] = args; try { return new RegExp(pat).test(String(s)); } catch { return false; } },
  point(_i, args) {
    // point(content, confidence) constructor (alternative to a point literal)
    return { __t: "point", content: String(args[0] ?? ""), confidence: clip01(Number(args[1] ?? 1)), interps: [] };
  },
  abs(_i, a) { return Math.abs(Number(a[0])); },
  round(_i, a) { const d = a[1] ?? 0; const f = 10 ** d; return Math.round(Number(a[0]) * f) / f; },
  min(_i, a) { return Math.min(...a.map(Number)); },
  max(_i, a) { return Math.max(...a.map(Number)); },
  sqrt(_i, a) { return Math.sqrt(Number(a[0])); },
  sum(_i, a) { const xs = Array.isArray(a[0]) ? a[0] : a; return xs.reduce((s, x) => s + Number(x), 0); },
  mean(_i, a) { const xs = Array.isArray(a[0]) ? a[0] : a; return xs.length ? xs.reduce((s, x) => s + Number(x), 0) / xs.length : 0; },
  lower(_i, a) { return String(a[0]).toLowerCase(); },
  upper(_i, a) { return String(a[0]).toUpperCase(); },
  str(_i, a) { return tbToString(a[0]); },
  num(_i, a) { return Number(a[0]); },
  range(_i, a) { const n = Number(a[0]); return Array.from({ length: n }, (_, i) => i); },
  append(_i, a) { if (Array.isArray(a[0])) a[0].push(a[1]); return a[0]; },

  // --- Polyglot tool resolver: run an inline Python snippet (Pyodide) ---
  async python(interp, args) {
    const code = String(args[0] ?? "");
    const { runPython } = await import("./python.js");
    const result = await runPython(code, interp.onStatus);
    return jsToTb(result);
  },

  // --- Polyglot tool resolver: delegate to a Python specialist file ---
  // trebuchet.delegate(specialist_filename, entry_function, data) -> structured result
  "trebuchet.delegate": async (interp, args) => {
    const specialist = String(args[0] ?? "");
    const entry = String(args[1] ?? "main");
    const data = tbToJS(args[2] ?? null);
    const src = interp.files[specialist];
    if (src == null) {
      throw new TbError(`trebuchet.delegate: specialist '${specialist}' not found among project files`);
    }
    const { runSpecialist } = await import("./python.js");
    const result = await runSpecialist(src, entry, data, interp.onStatus);
    return jsToTb(result);
  },

  // --- AI oracle resolvers (transformers.js, in-browser models) ---
  async summarize(interp, args) {
    const { summarize } = await import("./models.js");
    return await summarize(String(args[0] ?? ""), interp.onStatus);
  },
  async classify(interp, args) {
    const labels = Array.isArray(args[1]) ? args[1].map((x) => String(x)) : [];
    const { classify } = await import("./models.js");
    return jsToTb(await classify(String(args[0] ?? ""), labels, interp.onStatus));
  },
  async ask(interp, args) {
    const { ask } = await import("./models.js");
    return jsToTb(await ask(String(args[0] ?? ""), String(args[1] ?? ""), interp.onStatus));
  },
};

/* --------------------------- 6. ENTRY POINT ------------------------------- */

export async function run(source, opts = {}) {
  const result = { ok: false, output: [], propositions: [], points: [], value: null, error: null, ast: null };
  let interp;
  try {
    const tokens = lex(source);
    const ast = new Parser(tokens).parseProgram();
    result.ast = ast;
    interp = new Interpreter({ files: opts.files || {}, onStatus: opts.onStatus });
    await interp.run(ast);
    result.ok = true;
  } catch (e) {
    if (e instanceof ReturnSignal) { result.ok = true; }
    else result.error = { message: e.message, line: e.line ?? null };
  }
  if (interp) {
    result.output = interp.output;
    result.propositions = interp.propositions;
    result.points = interp.points;
  }
  return result;
}

export { tbToString, noisyOr };
