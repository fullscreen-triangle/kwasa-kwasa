/* =====================================================================
   charts.js -- d3 renderings for ```chart fences in an .ndo document.

   A chart fence is JSON, not code. The document says what to draw and
   which cell bindings to draw it from; this file draws it. Nothing here
   invents data: every value comes from the bindings the runtime wrote
   into the output block beneath the cell, so a chart cannot disagree
   with the run that produced it. If a binding is missing the chart says
   so, in place, rather than rendering an empty frame.

   Ported from the enzymology notebook's charts.js, same d3 v7 idiom,
   same dark palette, reduced to the four kinds these tutorials use.
   ===================================================================== */

const C = {
  c1: "#6d9fd0", c2: "#d0876a", c3: "#7fae8a", c4: "#a98ac0",
  dim: "#4a4a4a", grid: "#1c1c1c", axis: "#2e2e2e", fg: "#c2c2c2",
  ok: "#7fae8a", no: "#d0876a",
};
const SEQ = [C.c1, C.c2, C.c3, C.c4];

/* -- tooltip --------------------------------------------------------- */

let tipEl = null;
function tip() {
  if (!tipEl) {
    tipEl = document.createElement("div");
    tipEl.className = "tip";
    document.body.appendChild(tipEl);
  }
  return tipEl;
}
function showTip(ev, html) {
  const t = tip();
  t.innerHTML = html;
  t.style.left = ev.pageX + 12 + "px";
  t.style.top = ev.pageY - 10 + "px";
  t.style.opacity = 1;
}
function hideTip() { if (tipEl) tipEl.style.opacity = 0; }

/* -- frame ----------------------------------------------------------- */

function frame(host, title, opts) {
  opts = opts || {};
  const W = opts.width || 560, H = opts.height || 240;
  const m = Object.assign({ t: 14, r: 16, b: 38, l: 52 }, opts.margin || {});
  const box = document.createElement("div");
  box.className = "chart";
  if (title) {
    const h = document.createElement("p");
    h.className = "ct";
    h.textContent = title;
    box.appendChild(h);
  }
  host.appendChild(box);
  const svg = d3.select(box).append("svg")
    .attr("viewBox", `0 0 ${W} ${H}`)
    .attr("preserveAspectRatio", "xMidYMid meet");
  const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
  return { svg, g, W, H, m, iw: W - m.l - m.r, ih: H - m.t - m.b, box };
}

function axes(f, x, y, xlab, ylab, opts) {
  opts = opts || {};
  const xa = d3.axisBottom(x).ticks(opts.xticks || 5).tickSizeOuter(0);
  const ya = d3.axisLeft(y).ticks(opts.yticks || 4).tickSizeOuter(0);
  if (opts.xformat) xa.tickFormat(opts.xformat);
  if (opts.yformat) ya.tickFormat(opts.yformat);

  f.g.append("g").attr("class", "axis")
    .attr("transform", `translate(0,${f.ih})`).call(xa);
  f.g.append("g").attr("class", "axis").call(ya);

  if (xlab) {
    f.g.append("text").attr("class", "alab")
      .attr("x", f.iw / 2).attr("y", f.ih + 32)
      .attr("text-anchor", "middle").text(xlab);
  }
  if (ylab) {
    f.g.append("text").attr("class", "alab")
      .attr("transform", "rotate(-90)")
      .attr("x", -f.ih / 2).attr("y", -38)
      .attr("text-anchor", "middle").text(ylab);
  }
}

/* -- 1. ladder: what each rung closes of what is left ----------------- */

function ladderChart(host, spec, powers) {
  const target = spec.target;
  const f = frame(host, spec.title, spec);
  const n = powers.length;
  const x = d3.scaleLinear().domain([0, n]).range([0, f.iw]);
  const y = d3.scaleLinear().domain([0, 1]).range([f.ih, 0]);

  axes(f, x, y, "rungs applied", "composite power", {
    xticks: n, xformat: d3.format("d"),
  });

  // The composite after each rung: 1 - prod(1 - p).
  const pts = [0];
  let residual = 1;
  for (const p of powers) { residual *= 1 - p; pts.push(1 - residual); }

  if (typeof target === "number") {
    f.g.append("line").attr("class", "rule")
      .attr("x1", 0).attr("x2", f.iw)
      .attr("y1", y(target)).attr("y2", y(target))
      .attr("stroke", C.dim).attr("stroke-dasharray", "4 3");
    f.g.append("text").attr("class", "alab")
      .attr("x", f.iw).attr("y", y(target) - 6)
      .attr("text-anchor", "end").attr("fill", C.dim)
      .text("target " + target);
  }

  // Each rung as a band from the previous composite to the new one, so the
  // shrinking gap is the visible fact rather than the rising line.
  for (let i = 0; i < n; i++) {
    f.g.append("rect")
      .attr("x", x(i)).attr("width", x(i + 1) - x(i) - 2)
      .attr("y", y(pts[i + 1])).attr("height", y(pts[i]) - y(pts[i + 1]))
      .attr("fill", SEQ[i % SEQ.length]).attr("opacity", 0.55)
      .on("mousemove", (ev) => showTip(ev,
        `rung ${i + 1} &middot; power ${powers[i]}<br>closes ${
          fmt(pts[i + 1] - pts[i])} of the remaining ${fmt(1 - pts[i])}`))
      .on("mouseleave", hideTip);
  }

  const line = d3.line().x((d, i) => x(i)).y((d) => y(d));
  f.g.append("path").datum(pts).attr("class", "ln")
    .attr("d", line).attr("stroke", C.fg).attr("fill", "none")
    .attr("stroke-width", 1.6);

  f.g.selectAll("circle.pt").data(pts).enter().append("circle")
    .attr("cx", (d, i) => x(i)).attr("cy", (d) => y(d)).attr("r", 3)
    .attr("fill", C.fg)
    .on("mousemove", (ev, d) => showTip(ev, "composite " + fmt(d)))
    .on("mouseleave", hideTip);

  const final = pts[n];
  const clears = typeof target === "number" ? final >= target : null;
  if (clears !== null) {
    f.g.append("text").attr("class", "verdict")
      .attr("x", f.iw).attr("y", 12).attr("text-anchor", "end")
      .attr("fill", clears ? C.ok : C.no)
      .text(clears ? "clears" : "falls short by " + fmt(target - final));
  }
}

/* -- 2. cuts: every cut, with the minimum marked ---------------------- */

function cutsChart(host, spec, edges, sides, floor) {
  const f = frame(host, spec.title, spec);

  const weights = sides.map((side) => {
    const inside = new Set(side);
    let total = 0;
    for (const e of edges) {
      const a = inside.has(e.from), b = inside.has(e.to);
      if (a !== b) total += e.weight;
    }
    return { label: "{" + side.join(",") + "}", w: total };
  });

  const min = d3.min(weights, (d) => d.w);
  const x = d3.scaleBand().domain(weights.map((d) => d.label))
    .range([0, f.iw]).padding(0.28);
  const y = d3.scaleLinear()
    .domain([0, Math.max(d3.max(weights, (d) => d.w), floor || 0) * 1.15])
    .range([f.ih, 0]).nice();

  axes(f, x, y, "cut (source side)", "cut weight");

  f.g.selectAll("rect.bar").data(weights).enter().append("rect")
    .attr("x", (d) => x(d.label)).attr("width", x.bandwidth())
    .attr("y", (d) => y(d.w)).attr("height", (d) => f.ih - y(d.w))
    .attr("fill", (d) => (d.w === min ? C.c2 : C.dim))
    .attr("opacity", (d) => (d.w === min ? 0.95 : 0.5))
    .on("mousemove", (ev, d) => showTip(ev,
      d.label + "<br>weight " + fmt(d.w) +
      (d.w === min ? "<br><b>the minimum &mdash; this is sigma</b>" : "")))
    .on("mouseleave", hideTip);

  if (typeof floor === "number") {
    f.g.append("line").attr("class", "rule")
      .attr("x1", 0).attr("x2", f.iw)
      .attr("y1", y(floor)).attr("y2", y(floor))
      .attr("stroke", C.c4).attr("stroke-dasharray", "5 3");
    f.g.append("text").attr("class", "alab")
      .attr("x", 2).attr("y", y(floor) - 5)
      .attr("fill", C.c4).text("floor beta = " + floor);
  }

  const admissible = typeof floor === "number" ? min > floor : null;
  f.g.append("text").attr("class", "verdict")
    .attr("x", f.iw).attr("y", 12).attr("text-anchor", "end")
    .attr("fill", admissible ? C.ok : C.no)
    .text("sigma " + fmt(min) +
      (admissible === null ? "" : admissible ? " · admissible" : " · below floor"));
}

/* -- 3. contact graph: the two sources side by side ------------------- */

function graphChart(host, spec, edges, floor) {
  const f = frame(host, spec.title, spec);
  const nodes = [];
  const seen = new Set();
  for (const e of edges) {
    for (const id of [e.from, e.to]) {
      if (!seen.has(id)) { seen.add(id); nodes.push({ id }); }
    }
  }
  // Deterministic layout: source left, target right, the rest between.
  const order = spec.order || nodes.map((n) => n.id);
  const col = new Map();
  order.forEach((id, i) => col.set(id, i));
  const lanes = d3.max(order, (id) => col.get(id)) || 1;
  const x = d3.scaleLinear().domain([0, lanes]).range([20, f.iw - 20]);

  const mid = new Map();
  let above = true;
  for (const n of nodes) {
    const c = col.get(n.id) ?? 0;
    const edge = c === 0 || c === lanes;
    n.x = x(c);
    n.y = edge ? f.ih / 2 : (above ? f.ih * 0.24 : f.ih * 0.76);
    if (!edge) above = !above;
    mid.set(n.id, n);
  }

  const wex = d3.extent(edges, (e) => e.weight);
  const sw = d3.scaleLinear().domain(wex).range([0.7, 4.2]);
  const thin = typeof floor === "number";

  f.g.selectAll("line.edge").data(edges).enter().append("line")
    .attr("x1", (e) => mid.get(e.from).x).attr("y1", (e) => mid.get(e.from).y)
    .attr("x2", (e) => mid.get(e.to).x).attr("y2", (e) => mid.get(e.to).y)
    .attr("stroke", (e) => (thin && e.weight <= floor ? C.no : C.c1))
    .attr("stroke-width", (e) => sw(e.weight))
    .attr("opacity", 0.8)
    .on("mousemove", (ev, e) => showTip(ev,
      `${e.from} &rarr; ${e.to}<br>weight ${fmt(e.weight)}` +
      (thin && e.weight <= floor ? "<br><b>below the floor</b>" : "")))
    .on("mouseleave", hideTip);

  const g = f.g.selectAll("g.node").data(nodes).enter().append("g")
    .attr("transform", (n) => `translate(${n.x},${n.y})`);
  g.append("circle").attr("r", 13)
    .attr("fill", "#161616").attr("stroke", C.fg).attr("stroke-width", 1.2);
  g.append("text").attr("class", "nlab")
    .attr("text-anchor", "middle").attr("dy", 4)
    .attr("fill", C.fg).text((n) => n.id);
}

/* -- 4. stages: where each verdict is decided ------------------------- */

function stagesChart(host, spec, rows) {
  const f = frame(host, spec.title, Object.assign({ height: 280 }, spec));
  const stages = [];
  for (const r of rows) if (!stages.includes(r.stage)) stages.push(r.stage);

  const x = d3.scaleBand().domain(stages).range([0, f.iw]).padding(0.18);
  const y = d3.scaleBand()
    .domain(d3.range(d3.max(stages.map((s) =>
      rows.filter((r) => r.stage === s).length))))
    .range([0, f.ih]).padding(0.22);

  f.g.append("g").attr("class", "axis")
    .attr("transform", `translate(0,${f.ih})`)
    .call(d3.axisBottom(x).tickSizeOuter(0));

  // The dividing line: everything left of it is decided before the network.
  const cut = spec.before_request || 0;
  if (cut > 0 && cut < stages.length) {
    const xc = x(stages[cut]) - x.step() * x.paddingInner() / 2;
    f.g.append("line").attr("x1", xc).attr("x2", xc)
      .attr("y1", -4).attr("y2", f.ih)
      .attr("stroke", C.c4).attr("stroke-dasharray", "5 3");
    f.g.append("text").attr("class", "alab")
      .attr("x", xc - 6).attr("y", -6).attr("text-anchor", "end")
      .attr("fill", C.c4).text("no request issued");
    f.g.append("text").attr("class", "alab")
      .attr("x", xc + 6).attr("y", -6)
      .attr("fill", C.dim).text("request issued");
  }

  const byStage = new Map();
  for (const r of rows) {
    const k = r.stage;
    if (!byStage.has(k)) byStage.set(k, []);
    const i = byStage.get(k).length;
    byStage.get(k).push(r);
    const g = f.g.append("g")
      .attr("transform", `translate(${x(k)},${y(i)})`);
    g.append("rect")
      .attr("width", x.bandwidth()).attr("height", y.bandwidth())
      .attr("rx", 3)
      .attr("fill", r.name === "answer" ? C.ok : C.c2)
      .attr("opacity", r.name === "answer" ? 0.8 : 0.55)
      .on("mousemove", (ev) => showTip(ev,
        `<b>${r.name}</b><br>decided at the ${r.stage} stage<br>fix: ${r.fix}`))
      .on("mouseleave", hideTip);
    g.append("text").attr("class", "nlab")
      .attr("x", x.bandwidth() / 2).attr("y", y.bandwidth() / 2 + 4)
      .attr("text-anchor", "middle").attr("fill", "#0f0f0f")
      .text(r.name);
  }
}

/* -- number formatting, matching the notebook ------------------------- */

function fmt(v) {
  if (typeof v !== "number" || !isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e5 || a < 1e-4) return d3.format(".3e")(v);
  return d3.format(a < 1 ? ".4f" : ".3f")(v).replace(/\.?0+$/, "");
}

/* -- dispatch --------------------------------------------------------- */

/* Read a value out of the bindings the runtime wrote for a cell. `spec`
   names bindings rather than carrying data, so the chart is always of the
   run that just happened. A name that is not bound is an error shown in
   place -- never a silently empty chart. */
function pick(bindings, name) {
  if (!(name in bindings)) {
    throw new Error("no binding named '" + name + "' in the cell above");
  }
  return bindings[name];
}

function drawChart(host, spec, bindings) {
  switch (spec.chart) {
    case "ladder":
      return ladderChart(host, spec, pick(bindings, spec.powers));
    case "cuts":
      return cutsChart(host, spec,
        pick(bindings, spec.edges), pick(bindings, spec.sides),
        spec.floor === undefined ? undefined
          : (typeof spec.floor === "string" ? pick(bindings, spec.floor) : spec.floor));
    case "graph":
      return graphChart(host, spec, pick(bindings, spec.edges),
        spec.floor === undefined ? undefined
          : (typeof spec.floor === "string" ? pick(bindings, spec.floor) : spec.floor));
    case "stages":
      return stagesChart(host, spec, pick(bindings, spec.rows));
    default:
      throw new Error("unknown chart kind '" + spec.chart + "'");
  }
}
