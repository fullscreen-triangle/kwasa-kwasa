/* CodeMirror 6 language support for Turbulance.
 *
 * A StreamLanguage tokenizer keyed by the same keyword set the interpreter
 * uses. The token categories here (keyword / def / atom / string / number /
 * comment / operator) are deliberately the standard ones, so the same
 * classification maps directly onto a TextMate grammar when this is packaged
 * as a VS Code extension. */
import { StreamLanguage, LanguageSupport } from "@codemirror/language";

const KEYWORDS = new Set([
  "funxn", "item", "proposition", "motion", "support", "contradict",
  "inconclusive", "within", "given", "considering", "for", "each", "in",
  "while", "return", "ensure", "allow", "research", "cause", "point",
  "resolution", "resolve", "cycle", "drift", "flow", "roll", "until",
  "settled", "over", "on", "goal", "metacognitive", "import", "from", "as",
  "otherwise", "with_confidence", "and", "or", "not",
]);

// After one of these, the next identifier is a declared name (highlight as def).
const DECL = new Set([
  "funxn", "proposition", "motion", "point", "goal", "metacognitive",
  "evidence", "pattern", "item", "cause",
]);

const ATOMS = new Set(["true", "false", "none"]);

const parser = {
  startState() { return { expectName: false }; },
  token(stream, state) {
    if (stream.eatSpace()) return null;

    // comments
    if (stream.match("//") || stream.match("#")) { stream.skipToEnd(); return "comment"; }

    // strings (tolerate an unterminated string at end of line)
    if (stream.match(/^"(?:[^"\\]|\\.)*"?/)) return "string";

    // numbers
    if (stream.match(/^\d+(?:\.\d+)?/)) return "number";

    // identifiers / keywords
    if (stream.match(/^[A-Za-z_][A-Za-z0-9_]*/)) {
      const w = stream.current();
      const wasExpectName = state.expectName;
      state.expectName = DECL.has(w);
      if (ATOMS.has(w)) return "atom";
      if (KEYWORDS.has(w)) return "keyword";
      if (wasExpectName) return "def";
      return "variableName";
    }

    // operators
    if (stream.match(/^(?:==|!=|<=|>=|&&|\|\||\|>|=>)/)) return "operator";
    if (stream.match(/^[+\-*/<>=|!]/)) return "operator";

    stream.next();
    return null;
  },
};

const turbulanceLanguage = StreamLanguage.define(parser);

export function turbulance() {
  return new LanguageSupport(turbulanceLanguage);
}
