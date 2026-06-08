/* Pyodide-backed Python substrate for Turbulance's polyglot tool resolver.
 *
 * This is the browser realization of Paper B's "tool resolver": Turbulance
 * suspends to an external substrate (here, CPython on WebAssembly via Pyodide),
 * which does the heavy numeric work and returns structured data — JSON-in /
 * JSON-out, exactly the contract the real `trebuchet.delegate(...)` uses with a
 * Python specialist file. Loaded lazily (heavy), entirely client-side. */

const PYODIDE_VERSION = "0.26.2";
const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let pyodidePromise = null;
const loadedPkgs = new Set();

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error("Failed to load Pyodide from CDN"));
    document.head.appendChild(s);
  });
}

async function ensurePyodide(onStatus) {
  if (pyodidePromise) return pyodidePromise;
  pyodidePromise = (async () => {
    onStatus && onStatus("Loading Python (Pyodide ~7 MB)…");
    if (!globalThis.loadPyodide) await loadScript(`${PYODIDE_BASE}pyodide.js`);
    const py = await globalThis.loadPyodide({ indexURL: PYODIDE_BASE });
    onStatus && onStatus("Python ready.");
    return py;
  })();
  return pyodidePromise;
}

function detectPackages(src) {
  const pkgs = [];
  if (/\bimport\s+numpy\b|\bnumpy\b|\bnp\./.test(src)) pkgs.push("numpy");
  if (/\bscipy\b/.test(src)) pkgs.push("scipy");
  if (/\bpandas\b|\bpd\./.test(src)) pkgs.push("pandas");
  return pkgs;
}

async function ensurePackages(py, pkgs, onStatus) {
  const need = pkgs.filter((p) => !loadedPkgs.has(p));
  if (!need.length) return;
  onStatus && onStatus(`Loading Python packages: ${need.join(", ")}…`);
  await py.loadPackage(need);
  need.forEach((p) => loadedPkgs.add(p));
}

const toJsOpts = { dict_converter: Object.fromEntries };

/* Run an inline Python snippet; return its last expression as plain JS. */
export async function runPython(code, onStatus) {
  const py = await ensurePyodide(onStatus);
  await ensurePackages(py, detectPackages(code), onStatus);
  onStatus && onStatus("Running Python…");
  const res = await py.runPythonAsync(code);
  const js = res && res.toJs ? res.toJs(toJsOpts) : res;
  if (res && res.destroy) res.destroy();
  onStatus && onStatus(null);
  return js;
}

/* Load a Python specialist's source, call entry(data), return the result as JS.
 * `data` is a plain JS object; the result must be JSON-able (dict/list/scalars). */
export async function runSpecialist(src, entry, data, onStatus) {
  const py = await ensurePyodide(onStatus);
  await ensurePackages(py, detectPackages(src), onStatus);
  onStatus && onStatus(`Running ${entry}()…`);
  py.runPython(src);
  const fn = py.globals.get(entry);
  if (!fn) {
    throw new Error(`Python specialist has no function named '${entry}'`);
  }
  const pyData = py.toPy(data == null ? {} : data);
  let out;
  try {
    let r = fn(pyData);
    if (r && typeof r.then === "function") r = await r; // tolerate async specialists
    out = r && r.toJs ? r.toJs(toJsOpts) : r;
    if (r && r.destroy) r.destroy();
  } finally {
    if (pyData && pyData.destroy) pyData.destroy();
    if (fn && fn.destroy) fn.destroy();
  }
  onStatus && onStatus(null);
  return out;
}
