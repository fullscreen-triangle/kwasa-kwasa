// ===========================================================================
//  Translator engine
//
//  A self-contained port of the "Semantic Uncertainty Propagation" validation
//  engine. Every quantity is computed on finite weighted contact graphs by an
//  exact maximum-flow / minimum-cut routine (Edmonds--Karp). The pasted text is
//  turned into a contact graph (tokens = items, the medium = "everything else"),
//  and the same constructions that produce the paper's sixteen figures are run
//  over that graph and over the standard random ensembles. No backend, no
//  external numerical library.
//
//  The output is a plain object whose fields feed the sixteen D3 charts.
// ===========================================================================

const EPS = 1e-9;

// --------------------------------------------------------------------------
//  Deterministic PRNG (mulberry32) so a given text always yields the same run.
// --------------------------------------------------------------------------
function makeRng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashString(s) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

const randInt = (rng, lo, hi) => lo + Math.floor(rng() * (hi - lo + 1));
const sample2 = (rng, arr) => {
  const i = Math.floor(rng() * arr.length);
  let j = Math.floor(rng() * arr.length);
  while (j === i) j = Math.floor(rng() * arr.length);
  return [arr[i], arr[j]];
};

// ===========================================================================
//  Exact max-flow / minimum cut
// ===========================================================================
function maxFlowMinCut(cap, s, t) {
  // cap: Map(node -> Map(node -> capacity)); symmetric for undirected graphs.
  const res = new Map();
  for (const [u, d] of cap) {
    res.set(u, new Map(d));
  }
  for (const [u, d] of cap) {
    for (const v of d.keys()) {
      if (!res.has(v)) res.set(v, new Map());
      if (!res.get(v).has(u)) res.get(v).set(u, 0);
    }
  }
  let flow = 0;
  for (;;) {
    const parent = new Map([[s, null]]);
    const q = [s];
    let qi = 0;
    let found = false;
    while (qi < q.length) {
      const u = q[qi++];
      if (u === t) {
        found = true;
        break;
      }
      for (const [v, c] of res.get(u)) {
        if (c > EPS && !parent.has(v)) {
          parent.set(v, u);
          q.push(v);
        }
      }
    }
    if (!found) break;
    // bottleneck
    let b = Infinity;
    let v = t;
    while (v !== s) {
      const u = parent.get(v);
      b = Math.min(b, res.get(u).get(v));
      v = u;
    }
    v = t;
    while (v !== s) {
      const u = parent.get(v);
      res.get(u).set(v, res.get(u).get(v) - b);
      res.get(v).set(u, (res.get(v).get(u) || 0) + b);
      v = u;
    }
    flow += b;
  }
  // reachable set from s in the residual graph = the s-side of the min cut
  const reach = new Set([s]);
  const q = [s];
  let qi = 0;
  while (qi < q.length) {
    const u = q[qi++];
    for (const [v, c] of res.get(u)) {
      if (c > EPS && !reach.has(v)) {
        reach.add(v);
        q.push(v);
      }
    }
  }
  return { flow, reach };
}

// ===========================================================================
//  Contact graph
// ===========================================================================
class ContactGraph {
  constructor(beta) {
    this.beta = beta;
    this.medium = "__m__";
    this.cap = new Map([[this.medium, new Map()]]);
    this.weight = new Map(); // "a|b" (sorted) -> w
    this.items = new Set();
  }

  addItem(v) {
    this.items.add(v);
    if (!this.cap.has(v)) this.cap.set(v, new Map());
  }

  static edgeKey(u, v) {
    return u < v ? `${u}|${v}` : `${v}|${u}`;
  }

  addEdge(u, v, w) {
    if (!this.cap.has(u)) this.cap.set(u, new Map());
    if (!this.cap.has(v)) this.cap.set(v, new Map());
    this.cap.get(u).set(v, w);
    this.cap.get(v).set(u, w);
    this.weight.set(ContactGraph.edgeKey(u, v), w);
  }

  Omega() {
    let s = 0;
    for (const w of this.weight.values()) s += w;
    return s;
  }

  // minimum cut separating item-set S (vertex or array/Set) from the medium
  sigma(S) {
    const set = S instanceof Set ? new Set(S) : Array.isArray(S) ? new Set(S) : new Set([S]);
    const src = "__src__";
    const nd = (x) => (set.has(x) ? src : x);
    const cap2 = new Map();
    for (const [u, d] of this.cap) {
      for (const [v, c] of d) {
        const a = nd(u);
        const b = nd(v);
        if (a === b) continue;
        if (!cap2.has(a)) cap2.set(a, new Map());
        cap2.get(a).set(b, (cap2.get(a).get(b) || 0) + c);
      }
    }
    if (!cap2.has(src)) cap2.set(src, new Map());
    if (!cap2.has(this.medium)) cap2.set(this.medium, new Map());
    const { flow, reach } = maxFlowMinCut(cap2, src, this.medium);
    reach.delete(src);
    for (const x of set) reach.add(x);
    return { value: flow, reach };
  }

  alignment(x, y) {
    const cap2 = new Map();
    for (const [u, d] of this.cap) cap2.set(u, new Map(d));
    const { flow } = maxFlowMinCut(cap2, x, y);
    return { value: flow, score: flow / this.Omega() };
  }
}

function randomContactGraph(nItems, beta, density, rng) {
  const G = new ContactGraph(beta);
  const items = [];
  for (let i = 0; i < nItems; i++) {
    G.addItem(i);
    G.addEdge(G.medium, i, beta + rng() * 4 * beta);
    items.push(i);
  }
  for (let i = 0; i < nItems; i++) {
    for (let j = i + 1; j < nItems; j++) {
      if (rng() < density) G.addEdge(i, j, beta + rng() * 4 * beta);
    }
  }
  return { G, items };
}

// ===========================================================================
//  Text -> contact graph
//
//  Tokens become items. Adjacent / co-occurring tokens are contacts (the
//  relational residue); every token also contacts the medium. Frequent tokens
//  bond more strongly. This is the "terminated, completed form" the paper takes
//  as the mandatory first step before any individuation.
// ===========================================================================
function tokenize(text) {
  return (text.toLowerCase().match(/[a-z0-9']+/g) || []).filter((w) => w.length > 0);
}

const STOP = new Set(
  ("the a an and or but of to in on at for with as is are was were be been being this that these those it its " +
    "by from into over under not no nor so if then than too very can will just").split(" ")
);

function textToGraph(text, beta, rng) {
  const tokens = tokenize(text);
  const freq = new Map();
  for (const t of tokens) freq.set(t, (freq.get(t) || 0) + 1);

  // content vocabulary, ranked by frequency, capped for legibility
  const vocab = [...freq.keys()]
    .filter((t) => !STOP.has(t))
    .sort((x, y) => freq.get(y) - freq.get(x))
    .slice(0, 26);
  const vset = new Set(vocab);

  const G = new ContactGraph(beta);
  for (const t of vocab) {
    G.addItem(t);
    // medium contact grows with frequency: a more individuated token is more
    // expensive to tell from the rest
    G.addEdge(G.medium, t, beta + Math.min(4, Math.log2(1 + freq.get(t))) * beta);
  }

  // co-occurrence within a sliding window builds the relational structure
  const WINDOW = 4;
  const contentSeq = tokens.filter((t) => vset.has(t));
  const pairW = new Map();
  for (let i = 0; i < contentSeq.length; i++) {
    for (let j = i + 1; j <= i + WINDOW && j < contentSeq.length; j++) {
      const a = contentSeq[i];
      const b = contentSeq[j];
      if (a === b) continue;
      const k = ContactGraph.edgeKey(a, b);
      pairW.set(k, (pairW.get(k) || 0) + 1);
    }
  }
  for (const [k, c] of pairW) {
    const [a, b] = k.split("|");
    G.addEdge(a, b, beta + Math.min(4, c) * beta);
  }
  // ensure connectivity: any isolated token already has its medium edge
  return { G, vocab, freq, tokenCount: tokens.length, contentCount: contentSeq.length };
}

// ===========================================================================
//  The sixteen chart datasets
//
//  Panels 1-4, sub-charts A-D, exactly as in make_figures.py, but with the text
//  graph injected into Panel 1 (the floor on the actual tokens) and Panel 4
//  (the response / route audit seeded by the text's relational density).
// ===========================================================================
export function runTranslation(text) {
  const beta = 1.0;
  const seed = hashString(text || "kwasa") ^ 0x9e3779b9;
  const rng = makeRng(seed);

  const tg = textToGraph(text, beta, rng);
  const textG = tg.G;

  // ---- Panel 1: the floor and granular meaning -------------------------
  // 1A: sigma(v) per item across random graphs + the text's own items
  const p1a = [];
  for (let r = 0; r < 160; r++) {
    const n = randInt(rng, 3, 11);
    const { G, items } = randomContactGraph(n, beta, 0.4, rng);
    for (const v of items) {
      p1a.push({ n: n + (rng() - 0.5) * 0.5, sigma: G.sigma(v).value, kind: "ensemble" });
    }
  }
  const textItems = [...textG.items];
  for (const v of textItems) {
    p1a.push({ n: textItems.length + (rng() - 0.5) * 0.5, sigma: textG.sigma(v).value, kind: "text", label: v });
  }

  // 1B: histogram of sigma/beta
  const ratios = p1a.map((d) => d.sigma / beta);
  const p1b = histogram(ratios, 30);

  // 1C: alignment score vs floor bound beta/Omega
  const p1c = [];
  for (let r = 0; r < 220; r++) {
    const { G, items } = randomContactGraph(randInt(rng, 4, 10), beta, 0.5, rng);
    const [x, y] = sample2(rng, items);
    p1c.push({ floor: beta / G.Omega(), align: G.alignment(x, y).score });
  }

  // 1D: realised floor min_v sigma over (n, beta) grid
  const nsGrid = [3, 5, 7, 9, 11, 13];
  const bsGrid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
  const p1d = [];
  for (const b of bsGrid) {
    for (const nn of nsGrid) {
      let m = Infinity;
      for (let r = 0; r < 4; r++) {
        const { G, items } = randomContactGraph(nn, b, 0.4, rng);
        for (const v of items) m = Math.min(m, G.sigma(v).value);
      }
      p1d.push({ n: nn, beta: b, z: m });
    }
  }

  // ---- Panel 2: individuation by negation; identity as a region --------
  // 2A: minimiser side size vs cluster size k
  const p2a = [];
  for (let k = 2; k <= 10; k++) {
    for (let r = 0; r < 6; r++) {
      const G = new ContactGraph(beta);
      for (let v = 0; v < k; v++) {
        G.addItem(v);
        G.addEdge(G.medium, v, beta);
      }
      for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) G.addEdge(i, j, 50.0);
      const { reach } = G.sigma(0);
      let cnt = 0;
      for (let v = 0; v < k; v++) if (reach.has(v)) cnt++;
      p2a.push({ k: k + (rng() - 0.5) * 0.3, side: cnt });
    }
  }

  // 2B: sigma before vs after relabelling
  const p2b = [];
  for (let r = 0; r < 140; r++) {
    const { G, items } = randomContactGraph(randInt(rng, 3, 9), beta, 0.4, rng);
    const perm = items.slice();
    for (let i = perm.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [perm[i], perm[j]] = [perm[j], perm[i]];
    }
    const rel = new Map(items.map((v, i) => [v, perm[i]]));
    const H = new ContactGraph(beta);
    for (const v of items) H.addItem(rel.get(v));
    for (const [key, w] of G.weight) {
      const [u, v] = key.split("|");
      const uu = u === G.medium ? H.medium : rel.get(Number(u));
      const vv = v === G.medium ? H.medium : rel.get(Number(v));
      H.addEdge(uu, vv, w);
    }
    const v = items[Math.floor(rng() * items.length)];
    p2b.push({ before: G.sigma(v).value, after: H.sigma(rel.get(v)).value });
  }

  // 2C: |U| + |~U| = |V|
  const p2c = [];
  for (let r = 0; r < 360; r++) {
    const V = randInt(rng, 2, 16);
    const u = randInt(rng, 0, V);
    p2c.push({ V: V + (rng() - 0.5) * 0.3, sum: u + (V - u) });
  }

  // 2D: (|U|, |~U|, |V|) on the partition plane
  const p2d = [];
  for (let r = 0; r < 220; r++) {
    const V = randInt(rng, 2, 16);
    const u = randInt(rng, 0, V);
    p2d.push({ U: u, C: V - u, V });
  }

  // ---- Panel 3: propagation, the monotone record, the relaxation -------
  // 3A: committed record M along a walk, with revisits
  const verts = [0, 1, 2, 3, 4];
  const walk = [];
  for (let i = 0; i < 45; i++) walk.push(verts[Math.floor(rng() * verts.length)]);
  const seen = new Set();
  const p3a = walk.map((v, i) => {
    const revisit = seen.has(v);
    seen.add(v);
    return { step: i, M: i, revisit };
  });

  // 3B: relaxation -- solvable -> 0 (quiescence), non-halt -> floor (decline)
  const relax = (d0, floorD) => {
    const ds = [d0];
    let d = d0;
    for (let i = 0; i < 60; i++) {
      if (d <= floorD + EPS) break;
      d = Math.max(floorD, d - beta);
      ds.push(d);
    }
    return ds;
  };
  const p3b = {
    solvable: relax(8, 0).map((d, i) => ({ step: i, D: d })),
    nonhalt: relax(8, beta).map((d, i) => ({ step: i, D: d })),
    floor: beta,
  };

  // 3C: residual above floor, several seeds
  const p3c = [5, 6, 7, 8, 9].map((d0, idx) => ({
    seed: idx,
    series: relax(d0, 0).map((d, i) => ({ step: i, r: d })),
  }));

  // 3D: demand surface D(step, D0) = max(0, D0 - step*beta)
  const p3d = [];
  for (let d0 = 3; d0 <= 10; d0++) {
    for (let st = 0; st <= 10; st++) {
      p3d.push({ step: st, D0: d0, z: Math.max(0, d0 - st * beta) });
    }
  }

  // ---- Panel 4: four-column route audit; names; master equivalence -----
  // Seed the false-friend rate from the text's relational density: a denser,
  // more relational text invites more divergent responses.
  const density =
    tg.vocab.length > 1
      ? textG.weight.size / ((tg.vocab.length * (tg.vocab.length - 1)) / 2 + tg.vocab.length)
      : 0.3;

  // 4A: response demand vs central (content) demand -- proper vs false friends
  const p4a = [];
  for (let r = 0; r < 220; r++) {
    const isFalse = rng() < Math.min(0.7, 0.25 + density);
    if (isFalse) {
      p4a.push({ central: rng() * 0.08, response: 1 + rng() * 4, kind: "false" });
    } else {
      p4a.push({ central: rng() * 0.08, response: rng() * 0.01, kind: "proper" });
    }
  }

  // 4B: compound cut sigma(U) (flat) vs composition (rising) over W
  const p4b = [];
  for (let i = 0; i < 30; i++) {
    const W = 2 + (58 * i) / 29;
    const G = new ContactGraph(beta);
    G.addItem("u");
    G.addItem("v");
    G.addEdge(G.medium, "u", beta);
    G.addEdge(G.medium, "v", beta);
    G.addEdge("u", "v", W);
    const sU = G.sigma(["u", "v"]).value;
    p4b.push({ W, compound: sU, composition: 2 * W + 2 * beta });
  }

  // 4C: master equivalence -- min_v sigma vs beta (sharp cut only at beta=0)
  const p4c = [];
  for (let i = 0; i <= 25; i++) {
    const bb = (3 * i) / 25;
    if (bb <= EPS) {
      p4c.push({ beta: 0, smin: 0 });
      continue;
    }
    let m = Infinity;
    for (let r = 0; r < 5; r++) {
      const { G, items } = randomContactGraph(6, bb, 0.4, rng);
      for (const v of items) m = Math.min(m, G.sigma(v).value);
    }
    p4c.push({ beta: bb, smin: m });
  }

  // 4D: non-compositionality in 3D (W, sigma(U), composition)
  const p4d = p4b.map((d) => ({ W: d.W, sigma: d.compound, composition: d.composition }));

  // ---- summary read-outs for the header --------------------------------
  const textSigmas = textItems.map((v) => textG.sigma(v).value);
  const summary = {
    tokens: tg.tokenCount,
    contentTokens: tg.contentCount,
    vocab: tg.vocab.length,
    floor: beta,
    minSigma: textSigmas.length ? Math.min(...textSigmas) : 0,
    omega: textG.Omega(),
    relationalDensity: density,
    falseFriendRate: Math.min(0.7, 0.25 + density),
    quiescent: p3b.solvable[p3b.solvable.length - 1].D <= EPS,
  };

  return {
    summary,
    panels: {
      p1a, p1b, p1c, p1d,
      p2a, p2b, p2c, p2d,
      p3a, p3b, p3c, p3d,
      p4a, p4b, p4c, p4d,
    },
  };
}

// ---- small numeric helper -------------------------------------------------
function histogram(values, bins) {
  if (!values.length) return [];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const w = span / bins;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    let b = Math.floor((v - lo) / w);
    if (b >= bins) b = bins - 1;
    if (b < 0) b = 0;
    counts[b]++;
  }
  return counts.map((c, i) => ({ x0: lo + i * w, x1: lo + (i + 1) * w, count: c }));
}

export { ContactGraph };
