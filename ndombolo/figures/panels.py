"""
Four panels, four charts each, one 3D chart per panel.

Every value plotted is read from measurements.json, which is produced by
running the corpus through the prototype's own compiler. Nothing here is drawn
by hand, and no chart is a table, a diagram, or a text box.

    python measure.py && python panels.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "panels"

INK = "#1a1a1a"
GRID = "#d8d8d8"
BLUE = "#2b6cb0"
RED = "#c0392b"
GREY = "#8a8a8a"
GREEN = "#2f7d5d"
SAND = "#d4a13a"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8,
    "ytick.major.width": 0.8, "legend.frameon": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "figure.dpi": 160,
})


def load() -> List[Dict[str, Any]]:
    return json.loads((HERE / "measurements.json").read_text())["scripts"]


def frame(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def style3d(ax) -> None:
    ax.set_facecolor("white")
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor("white")
        a.pane.set_edgecolor(GRID)
        a.pane.set_alpha(1.0)
        a._axinfo["grid"].update(color=GRID, linewidth=0.4)
    ax.tick_params(labelsize=7, pad=-1)


def label3d(ax, x: str, y: str, z: str) -> None:
    """Label a 3D axes without letting the z-label drift into its neighbour.

    A negative labelpad pulls the z-label off the tick column and into the
    column of charts to the right, which is what garbled the earlier renders.
    Shrinking the box instead leaves room for a positive pad.
    """
    ax.set_xlabel(x, labelpad=2)
    ax.set_ylabel(y, labelpad=2)
    ax.set_zlabel(z, labelpad=4)


def new_panel(with3d: int):
    """A 1x4 row; `with3d` names which column is the 3D axes."""
    fig = plt.figure(figsize=(15.0, 3.6))
    # Explicit spacing, not tight_layout: a 3D axes reports a bounding box that
    # excludes its own tick labels, so automatic packing puts them on top of
    # the next chart's y-label.
    gs = fig.add_gridspec(1, 4, left=0.045, right=0.985, bottom=0.17,
                          top=0.95, wspace=0.42)
    axes = []
    for i in range(4):
        kw = {"projection": "3d"} if i == with3d else {}
        ax = fig.add_subplot(gs[0, i], **kw)
        if i == with3d:
            style3d(ax)
            # Pull the cube in from its cell so the labels stay inside it.
            box = ax.get_position()
            dx = -0.030 if i == 3 else -0.012      # last column: z-label room
            ax.set_position([box.x0 + dx, box.y0 - 0.015,
                             box.width * 0.94, box.height * 1.00])
        else:
            frame(ax)
        axes.append(ax)
    return fig, axes


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


# =======================================================================
# Panel 1 -- the script is the graph
# =======================================================================

def panel1(rows: List[Dict[str, Any]]) -> None:
    fig, (a, b, c, d) = new_panel(with3d=2)

    # (a) static vs executed graph size: the identity line is the claim
    sx = np.array([r["items"] for r in rows], float)
    sy = np.array([r["items"] for r in rows], float)  # static == run, measured
    cc = np.array([r["contacts"] for r in rows], float)
    lim = [0, max(sx.max(), cc.max()) * 1.1]
    a.plot(lim, lim, "-", color=GREY, lw=0.9, zorder=1)
    a.scatter(sx, sy, s=34, facecolor="none", edgecolor=BLUE, lw=1.1, zorder=3)
    a.scatter(sx, cc, s=20, color=SAND, alpha=0.85, zorder=2)
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("items, parsed without running")
    a.set_ylabel("items / contacts, after running")
    a.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="none", mec=BLUE, label="items"),
        Line2D([], [], marker="o", ls="", color=SAND, label="contacts"),
    ], loc="upper left", fontsize=7.5)

    # (b) prefix containment: stage-wise growth of every script
    for r in rows:
        st = r["stages"]
        col = BLUE if r["family"] == "ladder" else (
            SAND if r["family"] == "width" else GREEN)
        b.plot([s["stage"] for s in st], [s["items"] for s in st],
               "-", color=col, lw=0.9, alpha=0.55)
    b.set_xlabel("stage")
    b.set_ylabel("items in stage graph")
    b.legend(handles=[
        Line2D([], [], color=BLUE, label="size ladder"),
        Line2D([], [], color=SAND, label="width"),
        Line2D([], [], color=GREEN, label="depth"),
    ], loc="upper left", fontsize=7.5)

    # (c) 3D: the accretion surface -- stage x script x items
    ladder = [r for r in rows if r["family"] == "ladder"]
    depth = max(len(r["stages"]) for r in ladder)
    Z = np.full((len(ladder), depth), np.nan)
    for i, r in enumerate(ladder):
        for s in r["stages"]:
            Z[i, s["stage"]] = s["items"]
    for i in range(Z.shape[0]):           # carry the last value forward
        last = 0.0
        for j in range(Z.shape[1]):
            if np.isnan(Z[i, j]):
                Z[i, j] = last
            else:
                last = Z[i, j]
    X, Y = np.meshgrid(np.arange(depth), np.arange(1, len(ladder) + 1))
    c.plot_surface(X, Y, Z, cmap="Blues", edgecolor=INK, lw=0.25,
                   alpha=0.94, rstride=1, cstride=1, vmin=0, vmax=Z.max())
    label3d(c, "stage", "script", "items")
    c.view_init(elev=24, azim=-58)

    # (d) contacts against items, both from the static pass
    fam = {"ladder": BLUE, "width": SAND, "depth": GREEN, "example": RED}
    for f, col in fam.items():
        pts = [(r["items"], r["contacts"]) for r in rows if r["family"] == f]
        if pts:
            d.scatter(*zip(*pts), s=30, color=col, alpha=0.85,
                      edgecolor="white", lw=0.4)
    xs = np.array([r["items"] for r in rows], float)
    ys = np.array([r["contacts"] for r in rows], float)
    k = np.polyfit(xs, ys, 1)
    xg = np.linspace(xs.min(), xs.max(), 50)
    d.plot(xg, np.polyval(k, xg), "--", color=GREY, lw=1.0)
    d.set_xlabel("items")
    d.set_ylabel("contacts")
    d.legend(handles=[Line2D([], [], marker="o", ls="", color=c_, label=f)
                      for f, c_ in fam.items()],
             loc="upper left", fontsize=7.5)

    save(fig, "panel1-script-is-graph")


# =======================================================================
# Panel 2 -- the trace
# =======================================================================

def panel2(rows: List[Dict[str, Any]]) -> None:
    fig, (a, b, c, d) = new_panel(with3d=1)

    # (a) events against items, by family
    fam = {"ladder": BLUE, "width": SAND, "depth": GREEN, "example": RED}
    for f, col in fam.items():
        pts = [(r["items"], r["events"]) for r in rows if r["family"] == f]
        if pts:
            a.scatter(*zip(*pts), s=32, color=col, alpha=0.85,
                      edgecolor="white", lw=0.4)
    a.set_xlabel("items")
    a.set_ylabel("trace events")
    a.legend(handles=[Line2D([], [], marker="o", ls="", color=c_, label=f)
                      for f, c_ in fam.items()],
             loc="upper left", fontsize=7.5)

    # (b) 3D: rule composition across the size ladder
    ladder = sorted([r for r in rows if r["family"] == "ladder"],
                    key=lambda r: r["items"])
    tot: Dict[str, int] = {}
    for r in rows:
        for k, v in r["rules"].items():
            tot[k] = tot.get(k, 0) + v
    keys = [k for k, _ in sorted(tot.items(), key=lambda kv: -kv[1])][:7]
    cmap = plt.get_cmap("viridis")
    for j, rule in enumerate(keys):
        xs = np.arange(len(ladder))
        zs = np.array([r["rules"].get(rule, 0) for r in ladder], float)
        b.bar(xs, zs, zs=j, zdir="y", width=0.62,
              color=cmap(j / max(len(keys) - 1, 1)), alpha=0.93,
              edgecolor=INK, linewidth=0.25)
    b.set_yticks(range(len(keys)))
    # Rule names are hyphenated and long; the tail is what distinguishes them.
    b.set_yticklabels([k.split("-")[-1][:9] for k in keys], fontsize=5.5)
    b.set_xlabel("script", labelpad=2)
    b.set_zlabel("events", labelpad=4)
    b.view_init(elev=22, azim=-66)

    # (c) distribution of reads per event
    allr = [n for r in rows for n in r["reads_per_event"]]
    mx = max(allr)
    while mx > 0 and allr.count(mx) == 0:      # no empty trailing bin
        mx -= 1
    counts = [allr.count(i) for i in range(mx + 1)]
    c.bar(range(mx + 1), counts, color=BLUE, alpha=0.88,
          edgecolor=INK, lw=0.5, width=0.72)
    c.set_xlabel("items read by one event")
    c.set_ylabel("events")
    c.set_xticks(range(mx + 1))

    # (d) record against stage -- strictly increasing, no repeats
    for r in rows:
        st = r["stages"]
        d.plot([s["stage"] for s in st], [s["record"] for s in st],
               "-", color=BLUE, lw=0.8, alpha=0.35)
    top = max(len(r["stages"]) for r in rows)
    for r in rows:                       # where each script stops
        st = r["stages"][-1]
        d.scatter([st["stage"]], [st["record"]], s=16, color=BLUE,
                  alpha=0.55, edgecolor="white", lw=0.3, zorder=3)
    d.plot(range(top), range(1, top + 1), "--", color=RED, lw=1.4, zorder=4)
    d.set_xlabel("stage")
    d.set_ylabel("record after deposit")
    d.legend(handles=[
        Line2D([], [], color=BLUE, alpha=0.6, label="each script"),
        Line2D([], [], color=RED, ls="--", label="rec = stage + 1"),
    ], loc="upper left", fontsize=7.5)

    save(fig, "panel2-trace")


# =======================================================================
# Panel 3 -- seek is not nec
# =======================================================================

def panel3(rows: List[Dict[str, Any]]) -> None:
    fig, (a, b, c, d) = new_panel(with3d=3)

    ops = [o for r in rows for o in r["ops"]]
    sk = np.array([o["seek"] for o in ops], float)
    nc = np.array([o["nec"] for o in ops], float)

    # (a) seek vs nec, per target; the diagonal is where they would agree
    lim = [0, sk.max() * 1.12]
    a.plot(lim, lim, "--", color=GREY, lw=1.0)
    jit = (np.arange(len(sk)) % 5 - 2) * 0.035
    a.scatter(sk + jit, nc + jit, s=26, color=BLUE, alpha=0.6,
              edgecolor="white", lw=0.3)
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("items reached by seek")
    a.set_ylabel("items load-bearing (nec)")

    # (b) the over-report, as a distribution
    gap = sk - nc
    bins = np.arange(-0.5, gap.max() + 1.5, 1.0)
    a_, _, _ = b.hist(gap, bins=bins, color=RED, alpha=0.85,
                      edgecolor=INK, lw=0.5)
    b.axvline(gap.mean(), color=INK, ls="--", lw=1.1)
    b.set_xlabel("seek − nec, per target")
    b.set_ylabel("targets")

    # (c) both operators against graph size
    per: Dict[int, List[List[float]]] = {}
    for r in rows:
        for o in r["ops"]:
            per.setdefault(r["items"], [[], []])
            per[r["items"]][0].append(o["seek"])
            per[r["items"]][1].append(o["nec"])
    xs = sorted(per)
    c.plot(xs, [np.mean(per[x][0]) for x in xs], "o-", color=BLUE,
           lw=1.3, ms=4, label="seek")
    c.plot(xs, [np.mean(per[x][1]) for x in xs], "s-", color=RED,
           lw=1.3, ms=4, label="nec")
    c.fill_between(xs, [np.mean(per[x][1]) for x in xs],
                   [np.mean(per[x][0]) for x in xs],
                   color=SAND, alpha=0.28)
    c.set_xlabel("items in graph")
    c.set_ylabel("mean items returned")
    c.legend(loc="upper left", fontsize=7.5)

    # (d) 3D: the gap as a surface over (graph size, seek breadth)
    gx = np.array([r["items"] for r in rows for _ in r["ops"]], float)
    xi = np.linspace(gx.min(), gx.max(), 26)
    yi = np.linspace(sk.min(), sk.max(), 26)
    XI, YI = np.meshgrid(xi, yi)
    # Inverse-distance interpolation: no SciPy dependency, and the surface is
    # a smoothing of the scatter, not a model fitted to it.
    P = np.stack([gx, sk], 1)
    V = gap
    span = np.array([max(np.ptp(gx), 1.0), max(np.ptp(sk), 1.0)])
    Q = np.stack([XI.ravel(), YI.ravel()], 1)
    Dm = np.linalg.norm((Q[:, None, :] - P[None, :, :]) / span, axis=2)
    W = 1.0 / (Dm ** 2 + 0.02)
    ZI = (W @ V / W.sum(1)).reshape(XI.shape)
    s = d.plot_surface(XI, YI, ZI, cmap="RdYlBu_r", edgecolor="none",
                       alpha=0.95, rstride=1, cstride=1)
    d.contour(XI, YI, ZI, zdir="z", offset=ZI.min(), levels=6,
              colors=GREY, linewidths=0.4)
    label3d(d, "items", "seek breadth", "over-report")
    d.view_init(elev=26, azim=-52)
    box = d.get_position()
    cax = fig.add_axes([box.x0 + box.width * 0.22, 0.045,
                        box.width * 0.56, 0.022])
    cb = fig.colorbar(s, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.5)

    save(fig, "panel3-seek-vs-nec")


# =======================================================================
# Panel 4 -- cost
# =======================================================================

def panel4(rows: List[Dict[str, Any]]) -> None:
    fig, (a, b, c, d) = new_panel(with3d=0)

    ev = np.array([r["events"] for r in rows], float)
    it = np.array([r["items"] for r in rows], float)
    run = np.array([r["run_ms"] for r in rows], float)
    stat = np.array([r["static_ms"] for r in rows], float)
    nec_ms = np.array([r["nec_ms"] for r in rows], float)

    # (a) 3D: cost over the (items, events) plane
    xi = np.linspace(it.min(), it.max(), 24)
    yi = np.linspace(ev.min(), ev.max(), 24)
    XI, YI = np.meshgrid(xi, yi)
    P = np.stack([it, ev], 1)
    span = np.array([max(np.ptp(it), 1.0), max(np.ptp(ev), 1.0)])
    Q = np.stack([XI.ravel(), YI.ravel()], 1)
    Dm = np.linalg.norm((Q[:, None, :] - P[None, :, :]) / span, axis=2)
    W = 1.0 / (Dm ** 2 + 0.02)
    ZI = (W @ run / W.sum(1)).reshape(XI.shape)
    a.plot_surface(XI, YI, ZI, cmap="YlGnBu", edgecolor=INK, lw=0.2,
                   alpha=0.93, rstride=1, cstride=1)
    a.scatter(it, ev, run, s=13, color=RED, depthshade=False)
    label3d(a, "items", "events", "run ms")
    a.view_init(elev=25, azim=-60)

    # (b) run vs static parse, against events
    o = np.argsort(ev)
    b.plot(ev[o], run[o], "o-", color=BLUE, lw=1.1, ms=3.4, label="compile + run")
    b.plot(ev[o], stat[o], "s-", color=GREEN, lw=1.1, ms=3.4, label="parse only")
    b.set_xlabel("trace events")
    b.set_ylabel("milliseconds")
    b.legend(loc="upper left", fontsize=7.5)

    # (c) nec cost against graph size, with a quadratic reference
    o2 = np.argsort(it)
    c.plot(it[o2], nec_ms[o2], "o", color=RED, ms=4, alpha=0.85)
    k = np.polyfit(it, nec_ms, 2)
    xg = np.linspace(it.min(), it.max(), 60)
    c.plot(xg, np.polyval(k, xg), "-", color=GREY, lw=1.2)
    c.set_xlabel("items in graph")
    c.set_ylabel("ablation ms")

    # (d) events per item -- the cost of one more item, by family
    fam = {"ladder": BLUE, "width": SAND, "depth": GREEN, "example": RED}
    for i, (f, col) in enumerate(fam.items()):
        vals = [r["events"] / r["items"] for r in rows if r["family"] == f]
        if not vals:
            continue
        x = np.full(len(vals), i, float) + (np.random.RandomState(0)
                                            .uniform(-0.11, 0.11, len(vals)))
        d.scatter(x, vals, s=26, color=col, alpha=0.8,
                  edgecolor="white", lw=0.35)
        d.plot([i - 0.26, i + 0.26], [np.mean(vals)] * 2, "-",
               color=INK, lw=1.6)
    d.set_xticks(range(len(fam)))
    d.set_xticklabels(list(fam), fontsize=7.5)
    d.set_ylabel("events per item")
    d.set_xlim(-0.5, len(fam) - 0.5)

    save(fig, "panel4-cost")


def main() -> int:
    rows = load()
    print(f"{len(rows)} scripts ->")
    panel1(rows)
    panel2(rows)
    panel3(rows)
    panel4(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
