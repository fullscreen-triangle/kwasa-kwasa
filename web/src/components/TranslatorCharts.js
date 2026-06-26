// ===========================================================================
//  TranslatorCharts
//
//  Sixteen D3 charts (the four figure panels of "Semantic Uncertainty
//  Propagation", sub-charts A-D each), rendered as SVG into white cards. The
//  three-dimensional sub-charts (D of each panel) are drawn with a fixed
//  isometric projection so they read as 3-D without a WebGL dependency.
// ===========================================================================
import React, { useRef, useEffect } from "react";
import * as d3 from "d3";

const BLUE = "#2f6fb0";
const TEAL = "#1a9e8f";
const ORANGE = "#e08a2e";
const RED = "#cc3b3b";
const GREY = "#9aa3ad";

// generic hook: run a draw(svg, width, height) callback into a sized <svg>
function useChart(draw, deps) {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const width = el.clientWidth || 360;
    const height = el.clientHeight || 240;
    const svg = d3.select(el);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);
    draw(svg, width, height);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return ref;
}

function Card({ tag, title, subtitle, children }) {
  return (
    <div className="flex flex-col rounded-lg border border-dark/10 bg-white p-3 shadow-sm dark:border-light/15">
      <div className="mb-1 flex items-baseline gap-2">
        <span className="rounded bg-dark px-1.5 py-0.5 text-[10px] font-bold text-light dark:bg-light dark:text-dark">
          {tag}
        </span>
        <h3 className="text-[13px] font-semibold text-dark dark:text-light">{title}</h3>
      </div>
      {subtitle && (
        <p className="mb-1 text-[11px] leading-snug text-dark/55 dark:text-light/55">{subtitle}</p>
      )}
      <div className="relative w-full grow">{children}</div>
    </div>
  );
}

function Svg({ chartRef }) {
  return <svg ref={chartRef} className="h-[200px] w-full" preserveAspectRatio="xMidYMid meet" />;
}

// shared axis drawing
function axes(svg, x, y, m, w, h, xl, yl) {
  const g = svg.append("g");
  g.append("g")
    .attr("transform", `translate(0,${h - m.b})`)
    .call(d3.axisBottom(x).ticks(5).tickSize(3))
    .call((s) => s.selectAll("text").attr("font-size", 9).attr("fill", "#555"))
    .call((s) => s.selectAll(".domain,line").attr("stroke", "#bbb"));
  g.append("g")
    .attr("transform", `translate(${m.l},0)`)
    .call(d3.axisLeft(y).ticks(5).tickSize(3))
    .call((s) => s.selectAll("text").attr("font-size", 9).attr("fill", "#555"))
    .call((s) => s.selectAll(".domain,line").attr("stroke", "#bbb"));
  if (xl)
    svg.append("text").attr("x", (w + m.l) / 2).attr("y", h - 2).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", "#666").text(xl);
  if (yl)
    svg.append("text").attr("transform", "rotate(-90)").attr("x", -(h - m.b) / 2).attr("y", 11)
      .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", "#666").text(yl);
}

// fixed isometric projection for the "3-D" charts
function isoProjector(w, h, xr, yr, zr) {
  const cx = w * 0.5;
  const cy = h * 0.62;
  const scale = Math.min(w, h) * 0.42;
  const nx = (v) => (v - xr[0]) / (xr[1] - xr[0] || 1) - 0.5;
  const ny = (v) => (v - yr[0]) / (yr[1] - yr[0] || 1) - 0.5;
  const nz = (v) => (v - zr[0]) / (zr[1] - zr[0] || 1);
  const ax = Math.PI / 6; // 30 deg
  return (X, Y, Z) => {
    const x = nx(X);
    const y = ny(Y);
    const z = nz(Z);
    const px = (x - y) * Math.cos(ax);
    const py = (x + y) * Math.sin(ax) - z;
    return [cx + px * scale, cy + py * scale];
  };
}

function isoFrame(svg, proj, xr, yr, zr) {
  // draw the back floor square and vertical z axis for orientation
  const corners = [
    [xr[0], yr[0]],
    [xr[1], yr[0]],
    [xr[1], yr[1]],
    [xr[0], yr[1]],
  ].map(([a, b]) => proj(a, b, zr[0]));
  svg.append("polygon")
    .attr("points", corners.map((p) => p.join(",")).join(" "))
    .attr("fill", "#f6f7f9").attr("stroke", "#dfe3e8").attr("stroke-width", 1);
  const base = proj(xr[0], yr[1], zr[0]);
  const top = proj(xr[0], yr[1], zr[1]);
  svg.append("line").attr("x1", base[0]).attr("y1", base[1]).attr("x2", top[0]).attr("y2", top[1])
    .attr("stroke", "#cfd5db").attr("stroke-width", 1);
}

// ===========================================================================
//  PANEL 1
// ===========================================================================
function P1A({ data }) {
  const ref = useChart((svg, w, h) => {
    const m = { l: 34, r: 8, t: 8, b: 24 };
    const x = d3.scaleLinear().domain(d3.extent(data, (d) => d.n)).nice().range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, d3.max(data, (d) => d.sigma) * 1.05]).range([h - m.b, m.t]);
    axes(svg, x, y, m, w, h, "graph size n", "σ(v)");
    svg.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y(1)).attr("y2", y(1))
      .attr("stroke", RED).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.4);
    svg.append("g").selectAll("circle").data(data).join("circle")
      .attr("cx", (d) => x(d.n)).attr("cy", (d) => y(d.sigma))
      .attr("r", (d) => (d.kind === "text" ? 3 : 1.6))
      .attr("fill", (d) => (d.kind === "text" ? ORANGE : BLUE))
      .attr("opacity", (d) => (d.kind === "text" ? 0.95 : 0.45));
  }, [data]);
  return <Svg chartRef={ref} />;
}

function P1B({ data }) {
  const ref = useChart((svg, w, h) => {
    const m = { l: 30, r: 8, t: 8, b: 24 };
    const x = d3.scaleLinear().domain([d3.min(data, (d) => d.x0), d3.max(data, (d) => d.x1)]).range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, d3.max(data, (d) => d.count)]).nice().range([h - m.b, m.t]);
    axes(svg, x, y, m, w, h, "σ(v) / β", "count");
    svg.append("g").selectAll("rect").data(data).join("rect")
      .attr("x", (d) => x(d.x0) + 0.5).attr("width", (d) => Math.max(0.5, x(d.x1) - x(d.x0) - 1))
      .attr("y", (d) => y(d.count)).attr("height", (d) => y(0) - y(d.count))
      .attr("fill", TEAL).attr("opacity", 0.85);
    svg.append("line").attr("x1", x(1)).attr("x2", x(1)).attr("y1", m.t).attr("y2", h - m.b)
      .attr("stroke", RED).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.4);
  }, [data]);
  return <Svg chartRef={ref} />;
}

function P1C({ data }) {
  const ref = useChart((svg, w, h) => {
    const m = { l: 38, r: 8, t: 8, b: 24 };
    const mx = d3.max(data, (d) => Math.max(d.floor, d.align));
    const x = d3.scaleLinear().domain([0, mx]).nice().range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, mx]).nice().range([h - m.b, m.t]);
    axes(svg, x, y, m, w, h, "floor β/Ω", "alignment a");
    svg.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(mx)).attr("y2", y(mx))
      .attr("stroke", RED).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.4);
    svg.append("g").selectAll("circle").data(data).join("circle")
      .attr("cx", (d) => x(d.floor)).attr("cy", (d) => y(d.align)).attr("r", 1.8)
      .attr("fill", BLUE).attr("opacity", 0.45);
  }, [data]);
  return <Svg chartRef={ref} />;
}

function P1D({ data }) {
  const ref = useChart((svg, w, h) => {
    const xr = d3.extent(data, (d) => d.n);
    const yr = d3.extent(data, (d) => d.beta);
    const zr = [0, d3.max(data, (d) => d.z)];
    const proj = isoProjector(w, h, xr, yr, zr);
    isoFrame(svg, proj, xr, yr, zr);
    const color = d3.scaleSequential(d3.interpolateViridis).domain(zr);
    // sort back-to-front for painter's order
    const sorted = data.slice().sort((a, b) => a.n + a.beta - (b.n + b.beta));
    svg.append("g").selectAll("line").data(sorted).join("line")
      .attr("x1", (d) => proj(d.n, d.beta, zr[0])[0]).attr("y1", (d) => proj(d.n, d.beta, zr[0])[1])
      .attr("x2", (d) => proj(d.n, d.beta, d.z)[0]).attr("y2", (d) => proj(d.n, d.beta, d.z)[1])
      .attr("stroke", "#cbd2da").attr("stroke-width", 1);
    svg.append("g").selectAll("circle").data(sorted).join("circle")
      .attr("cx", (d) => proj(d.n, d.beta, d.z)[0]).attr("cy", (d) => proj(d.n, d.beta, d.z)[1])
      .attr("r", 3).attr("fill", (d) => color(d.z)).attr("stroke", "#fff").attr("stroke-width", 0.5);
    isoLabels(svg, w, h, "n", "β", "min σ");
  }, [data]);
  return <Svg chartRef={ref} />;
}

// ===========================================================================
//  PANEL 2
// ===========================================================================
function P2A({ data }) {
  const ref = useChart((svg, w, h) => {
    const m = { l: 30, r: 8, t: 8, b: 24 };
    const x = d3.scaleLinear().domain([1.5, 10.5]).range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, 11]).range([h - m.b, m.t]);
    axes(svg, x, y, m, w, h, "cluster size k", "|S*(v)|");
    svg.append("line").attr("x1", x(2)).attr("y1", y(2)).attr("x2", x(10)).attr("y2", y(10))
      .attr("stroke", RED).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.4);
    svg.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y(1)).attr("y2", y(1))
      .attr("stroke", GREY).attr("stroke-dasharray", "1 3");
    svg.append("g").selectAll("circle").data(data).join("circle")
      .attr("cx", (d) => x(d.k)).attr("cy", (d) => y(d.side)).attr("r", 2.4)
      .attr("fill", BLUE).attr("opacity", 0.55);
  }, [data]);
  return <Svg chartRef={ref} />;
}

function P2B({ data }) {
  const ref = useChart((svg, w, h) => {
    const m = { l: 34, r: 8, t: 8, b: 24 };
    const mx = d3.max(data, (d) => Math.max(d.before, d.after));
    const x = d3.scaleLinear().domain([0, mx]).nice().range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, mx]).nice().range([h - m.b, m.t]);
    axes(svg, x, y, m, w, h, "σ before", "σ after relabel");
    svg.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(mx)).attr("y2", y(mx))
      .attr("stroke", RED).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.4);
    svg.append("g").selectAll("circle").data(data).join("circle")
      .attr("cx", (d) => x(d.before)).attr("cy", (d) => y(d.after)).attr("r", 2)
      .attr("fill", TEAL).attr("opacity", 0.6);
  }, [data]);
  return <Svg chartRef={ref} />;
}

function P2C({ data }) {
  const ref = useChart((svg, w, h) => {
    const m = { l: 30, r: 8, t: 8, b: 24 };
    const x = d3.scaleLinear().domain([0, 16]).range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, 16]).range([h - m.b, m.t]);
    axes(svg, x, y, m, w, h, "|V|", "|U|+|∁U|");
    svg.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(16)).attr("y2", y(16))
      .attr("stroke", RED).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.4);
    svg.append("g").selectAll("circle").data(data).join("circle")
      .attr("cx", (d) => x(d.V)).attr("cy", (d) => y(d.sum)).attr("r", 2)
      .attr("fill", TEAL).attr("opacity", 0.5);
  }, [data]);
  return <Svg chartRef={ref} />;
}

function P2D({ data }) {
  const ref = useChart((svg, w, h) => {
    const xr = [0, 16], yr = [0, 16], zr = [0, 16];
    const proj = isoProjector(w, h, xr, yr, zr);
    isoFrame(svg, proj, xr, yr, zr);
    const color = d3.scaleSequential(d3.interpolatePlasma).domain(zr);
    const sorted = data.slice().sort((a, b) => a.U + a.C - (b.U + b.C));
    svg.append("g").selectAll("circle").data(sorted).join("circle")
      .attr("cx", (d) => proj(d.U, d.C, d.V)[0]).attr("cy", (d) => proj(d.U, d.C, d.V)[1])
      .attr("r", 2.6).attr("fill", (d) => color(d.V)).attr("opacity", 0.8);
    isoLabels(svg, w, h, "|U|", "|∁U|", "|V|");
  }, [data]);
  return <Svg chartRef={ref} />;
}

// ===========================================================================
//  PANEL 3
// ===========================================================================
function P3A({ data }) {
  const ref = useChart((svg, w, h) => {
    const m = { l: 30, r: 8, t: 8, b: 24 };
    const x = d3.scaleLinear().domain(d3.extent(data, (d) => d.step)).range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, d3.max(data, (d) => d.M)]).range([h - m.b, m.t]);
    axes(svg, x, y, m, w, h, "step", "record M");
    const line = d3.line().x((d) => x(d.step)).y((d) => y(d.M)).curve(d3.curveStepAfter);
    svg.append("path").datum(data).attr("fill", "none").attr("stroke", BLUE).attr("stroke-width", 1.6).attr("d", line);
    svg.append("g").selectAll("circle").data(data.filter((d) => d.revisit)).join("circle")
      .attr("cx", (d) => x(d.step)).attr("cy", (d) => y(d.M)).attr("r", 2.6).attr("fill", RED);
  }, [data]);
  return <Svg chartRef={ref} />;
}

function P3B({ data }) {
  const ref = useChart((svg, w, h) => {
    const m = { l: 30, r: 8, t: 8, b: 24 };
    const all = [...data.solvable, ...data.nonhalt];
    const x = d3.scaleLinear().domain(d3.extent(all, (d) => d.step)).range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, d3.max(all, (d) => d.D)]).range([h - m.b, m.t]);
    axes(svg, x, y, m, w, h, "relaxation step", "cross-demand D");
    svg.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y(data.floor)).attr("y2", y(data.floor))
      .attr("stroke", RED).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.2);
    const line = d3.line().x((d) => x(d.step)).y((d) => y(d.D));
    svg.append("path").datum(data.nonhalt).attr("fill", "none").attr("stroke", ORANGE).attr("stroke-width", 1.8).attr("d", line);
    svg.append("path").datum(data.solvable).attr("fill", "none").attr("stroke", TEAL).attr("stroke-width", 1.8).attr("d", line);
    svg.append("g").selectAll("circle").data(data.solvable).join("circle")
      .attr("cx", (d) => x(d.step)).attr("cy", (d) => y(d.D)).attr("r", 2).attr("fill", TEAL);
  }, [data]);
  return <Svg chartRef={ref} />;
}

function P3C({ data }) {
  const ref = useChart((svg, w, h) => {
    const m = { l: 30, r: 8, t: 8, b: 24 };
    const all = data.flatMap((s) => s.series);
    const x = d3.scaleLinear().domain(d3.extent(all, (d) => d.step)).range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, d3.max(all, (d) => d.r)]).range([h - m.b, m.t]);
    axes(svg, x, y, m, w, h, "step", "residual above floor");
    svg.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y(0)).attr("y2", y(0))
      .attr("stroke", RED).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.2);
    const line = d3.line().x((d) => x(d.step)).y((d) => y(d.r));
    svg.append("g").selectAll("path").data(data).join("path")
      .attr("fill", "none").attr("stroke", BLUE).attr("stroke-width", 1.3).attr("opacity", 0.7)
      .attr("d", (s) => line(s.series));
  }, [data]);
  return <Svg chartRef={ref} />;
}

function P3D({ data }) {
  const ref = useChart((svg, w, h) => {
    const xr = d3.extent(data, (d) => d.step);
    const yr = d3.extent(data, (d) => d.D0);
    const zr = [0, d3.max(data, (d) => d.z)];
    const proj = isoProjector(w, h, xr, yr, zr);
    isoFrame(svg, proj, xr, yr, zr);
    const color = d3.scaleSequential(d3.interpolateMagma).domain([zr[1], zr[0]]);
    // build quads over the (step, D0) grid for a surface look
    const steps = [...new Set(data.map((d) => d.step))].sort((a, b) => a - b);
    const d0s = [...new Set(data.map((d) => d.D0))].sort((a, b) => a - b);
    const at = (s, d) => data.find((p) => p.step === s && p.D0 === d);
    const quads = [];
    for (let i = 0; i < steps.length - 1; i++) {
      for (let j = 0; j < d0s.length - 1; j++) {
        const a = at(steps[i], d0s[j]);
        const b = at(steps[i + 1], d0s[j]);
        const c = at(steps[i + 1], d0s[j + 1]);
        const e = at(steps[i], d0s[j + 1]);
        if (a && b && c && e) quads.push([a, b, c, e]);
      }
    }
    quads.sort((q1, q2) => {
      const s1 = d3.mean(q1, (p) => p.step + p.D0);
      const s2 = d3.mean(q2, (p) => p.step + p.D0);
      return s1 - s2;
    });
    svg.append("g").selectAll("polygon").data(quads).join("polygon")
      .attr("points", (q) => q.map((p) => proj(p.step, p.D0, p.z).join(",")).join(" "))
      .attr("fill", (q) => color(d3.mean(q, (p) => p.z)))
      .attr("stroke", "#ffffff").attr("stroke-width", 0.4).attr("opacity", 0.95);
    isoLabels(svg, w, h, "step", "D₀", "demand");
  }, [data]);
  return <Svg chartRef={ref} />;
}

// ===========================================================================
//  PANEL 4
// ===========================================================================
function P4A({ data }) {
  const ref = useChart((svg, w, h) => {
    const m = { l: 34, r: 8, t: 8, b: 24 };
    const x = d3.scaleLinear().domain([0, d3.max(data, (d) => d.central)]).nice().range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, d3.max(data, (d) => d.response)]).nice().range([h - m.b, m.t]);
    axes(svg, x, y, m, w, h, "central demand", "response demand");
    svg.append("g").selectAll("circle").data(data).join("circle")
      .attr("cx", (d) => x(d.central)).attr("cy", (d) => y(d.response)).attr("r", 2.2)
      .attr("fill", (d) => (d.kind === "false" ? ORANGE : BLUE)).attr("opacity", 0.7);
    legend(svg, w, [["proper", BLUE], ["false friend", ORANGE]]);
  }, [data]);
  return <Svg chartRef={ref} />;
}

function P4B({ data }) {
  const ref = useChart((svg, w, h) => {
    const m = { l: 34, r: 8, t: 8, b: 24 };
    const x = d3.scaleLinear().domain(d3.extent(data, (d) => d.W)).range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, d3.max(data, (d) => d.composition)]).nice().range([h - m.b, m.t]);
    axes(svg, x, y, m, w, h, "internal weight W", "cut weight");
    const lc = d3.line().x((d) => x(d.W)).y((d) => y(d.composition));
    const lu = d3.line().x((d) => x(d.W)).y((d) => y(d.compound));
    svg.append("path").datum(data).attr("fill", "none").attr("stroke", ORANGE).attr("stroke-width", 1.8).attr("d", lc);
    svg.append("path").datum(data).attr("fill", "none").attr("stroke", BLUE).attr("stroke-width", 2).attr("d", lu);
    legend(svg, w, [["compound σ(U)", BLUE], ["composition", ORANGE]]);
  }, [data]);
  return <Svg chartRef={ref} />;
}

function P4C({ data }) {
  const ref = useChart((svg, w, h) => {
    const m = { l: 34, r: 8, t: 8, b: 24 };
    const x = d3.scaleLinear().domain([0, 3]).range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, d3.max(data, (d) => d.smin) || 3]).nice().range([h - m.b, m.t]);
    axes(svg, x, y, m, w, h, "floor β", "min σ");
    svg.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(3)).attr("y2", y(3))
      .attr("stroke", RED).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.2);
    svg.append("g").selectAll("circle").data(data).join("circle")
      .attr("cx", (d) => x(d.beta)).attr("cy", (d) => y(d.smin)).attr("r", 2.4)
      .attr("fill", (d) => (d.beta < 1e-6 ? RED : TEAL)).attr("opacity", 0.85);
  }, [data]);
  return <Svg chartRef={ref} />;
}

function P4D({ data }) {
  const ref = useChart((svg, w, h) => {
    const xr = d3.extent(data, (d) => d.W);
    const yr = [0, d3.max(data, (d) => d.sigma) * 1.4 || 3];
    const zr = [0, d3.max(data, (d) => d.composition)];
    const proj = isoProjector(w, h, xr, yr, zr);
    isoFrame(svg, proj, xr, yr, zr);
    const color = d3.scaleSequential(d3.interpolatePlasma).domain(zr);
    // ridge of the flat compound cut along z=0
    svg.append("path").datum(data)
      .attr("fill", "none").attr("stroke", "#c2c8d0").attr("stroke-width", 1)
      .attr("d", d3.line().x((d) => proj(d.W, d.sigma, 0)[0]).y((d) => proj(d.W, d.sigma, 0)[1]));
    svg.append("g").selectAll("circle").data(data).join("circle")
      .attr("cx", (d) => proj(d.W, d.sigma, d.composition)[0])
      .attr("cy", (d) => proj(d.W, d.sigma, d.composition)[1])
      .attr("r", 2.8).attr("fill", (d) => color(d.composition)).attr("opacity", 0.85);
    isoLabels(svg, w, h, "W", "σ(U)", "composition");
  }, [data]);
  return <Svg chartRef={ref} />;
}

// ---- small chart helpers --------------------------------------------------
function legend(svg, w, items) {
  const g = svg.append("g").attr("transform", `translate(${w - 96},10)`);
  items.forEach(([label, color], i) => {
    g.append("rect").attr("x", 0).attr("y", i * 13).attr("width", 8).attr("height", 8).attr("fill", color).attr("rx", 1);
    g.append("text").attr("x", 12).attr("y", i * 13 + 7).attr("font-size", 9).attr("fill", "#555").text(label);
  });
}

function isoLabels(svg, w, h, xl, yl, zl) {
  svg.append("text").attr("x", w * 0.78).attr("y", h - 8).attr("font-size", 9).attr("fill", "#888").text(xl);
  svg.append("text").attr("x", w * 0.12).attr("y", h - 8).attr("font-size", 9).attr("fill", "#888").text(yl);
  svg.append("text").attr("x", 6).attr("y", 14).attr("font-size", 9).attr("fill", "#888").text(zl);
}

// ===========================================================================
//  Layout: the sixteen cards grouped into four panels
// ===========================================================================
const PANELS = [
  {
    title: "Panel 1 · The floor and granular meaning",
    charts: [
      { tag: "1A", title: "Separation cost vs size", sub: "σ(v) ≥ β on every item; orange = your text's tokens", C: P1A, key: "p1a" },
      { tag: "1B", title: "Cost distribution", sub: "σ(v)/β supported at and above 1", C: P1B, key: "p1b" },
      { tag: "1C", title: "Alignment vs floor", sub: "a ≥ β/Ω: no two units perfectly aligned", C: P1C, key: "p1c" },
      { tag: "1D", title: "Realised floor surface", sub: "min σ over (n, β), strictly positive", C: P1D, key: "p1d", three: true },
    ],
  },
  {
    title: "Panel 2 · Individuation by negation; identity as a region",
    charts: [
      { tag: "2A", title: "Identity is a region", sub: "minimiser side |S*(v)| = k > 1", C: P2A, key: "p2a" },
      { tag: "2B", title: "Relabelling invariance", sub: "σ unchanged under relabelling", C: P2B, key: "p2b" },
      { tag: "2C", title: "Negation partition", sub: "|U|+|∁U| = |V| exactly", C: P2C, key: "p2c" },
      { tag: "2D", title: "Partition plane", sub: "(|U|,|∁U|,|V|) on one plane", C: P2D, key: "p2d", three: true },
    ],
  },
  {
    title: "Panel 3 · Propagation, the monotone record, the relaxation",
    charts: [
      { tag: "3A", title: "Monotone record", sub: "M strictly up; red = revisited position", C: P3A, key: "p3a" },
      { tag: "3B", title: "Quiescence vs decline", sub: "teal → 0 (proper); orange plateaus (decline)", C: P3B, key: "p3b" },
      { tag: "3C", title: "Residual descent", sub: "several seeds, each to the floor", C: P3C, key: "p3c" },
      { tag: "3D", title: "Demand surface", sub: "D(step, D₀) descends to the floor", C: P3D, key: "p3d", three: true },
    ],
  },
  {
    title: "Panel 4 · Route audit, names, the master equivalence",
    charts: [
      { tag: "4A", title: "Route audit", sub: "false friends: low content gap, high response gap", C: P4A, key: "p4a" },
      { tag: "4B", title: "Non-compositionality", sub: "σ(U) flat while composition rises", C: P4B, key: "p4b" },
      { tag: "4C", title: "Master equivalence", sub: "sharp cut only at β = 0", C: P4C, key: "p4c" },
      { tag: "4D", title: "Names in 3-D", sub: "(W, σ(U), composition): the flat ridge", C: P4D, key: "p4d", three: true },
    ],
  },
];

export default function TranslatorCharts({ result }) {
  const p = result.panels;
  return (
    <div className="flex flex-col gap-8">
      {PANELS.map((panel) => (
        <section key={panel.title}>
          <h2 className="mb-3 border-b border-dark/10 pb-1 text-sm font-bold uppercase tracking-wide text-dark/70 dark:border-light/15 dark:text-light/70">
            {panel.title}
          </h2>
          <div className="grid grid-cols-4 gap-4 lg:grid-cols-2 sm:grid-cols-1">
            {panel.charts.map((c) => (
              <Card key={c.tag} tag={c.tag + (c.three ? " · 3D" : "")} title={c.title} subtitle={c.sub}>
                <c.C data={p[c.key]} />
              </Card>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
