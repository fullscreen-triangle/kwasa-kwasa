import { run } from "./src/lib/turbulance/index.js";
import { examples } from "./src/lib/turbulance/examples.js";

let pass = 0, fail = 0, skipped = 0;
for (const ex of examples) {
  console.log("\n================================================");
  console.log(ex.title);
  console.log("------------------------------------------------");
  if (ex.files) {
    // Pyodide-backed examples need a browser; skip in the Node harness.
    skipped++;
    console.log("  (skipped — needs the browser/Pyodide)");
    continue;
  }
  const r = await run(ex.code);
  if (r.error) {
    fail++;
    console.log("  ERROR:", r.error.message, r.error.line ? `(line ${r.error.line})` : "");
    continue;
  }
  pass++;
  for (const line of r.output) console.log("  | " + line);
  for (const p of r.propositions) {
    console.log(`  * proposition ${p.name}: ${p.verdict} (score ${p.score.toFixed(3)})`);
    for (const m of p.motions) console.log(`      - ${m.name}: ${m.verdict} (${m.score.toFixed(3)}) — ${m.desc}`);
  }
}
console.log("\n================================================");
console.log(`RESULT: ${pass} ran, ${fail} errored, ${skipped} skipped`);
