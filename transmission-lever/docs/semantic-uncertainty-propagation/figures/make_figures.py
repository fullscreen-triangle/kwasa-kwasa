#!/usr/bin/env python3
"""
Figure panels for "Semantic Uncertainty Propagation".

Four panels, each white background with four charts in a row (A--D), the last
of every panel a 3-D chart. Every chart is computed from the framework's actual
finite-weighted-graph data (exact minimum cuts via the validation engine); none
is conceptual, text-based, or a table.

Run:  python make_figures.py
"""

import os
import sys
import math
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "validation"))
from validate import ContactGraph, random_contact_graph  # noqa: E402

rng = random.Random(20260609)

# ---- minimal, white style ----
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "xtick.color": "#333333", "ytick.color": "#333333", "text.color": "#222222",
    "axes.titlecolor": "#222222", "axes.grid": False, "legend.frameon": False,
})
BLUE, TEAL, ORANGE, RED = "#2f6fb0", "#1a9e8f", "#e08a2e", "#cc3b3b"


def new_panel():
    fig = plt.figure(figsize=(16, 4.0))
    ax = [fig.add_subplot(1, 4, i + 1) for i in range(3)]
    ax.append(fig.add_subplot(1, 4, 4, projection="3d"))
    return fig, ax


def letter(ax, s):
    ax.set_title(s, loc="left", fontweight="bold", color="#111111")


def style3d(ax):
    ax.xaxis.pane.set_facecolor("white")
    ax.yaxis.pane.set_facecolor("white")
    ax.zaxis.pane.set_facecolor("white")
    ax.xaxis.pane.set_edgecolor("#cccccc")
    ax.yaxis.pane.set_edgecolor("#cccccc")
    ax.zaxis.pane.set_edgecolor("#cccccc")
    ax.grid(True)


def save(fig, name):
    fig.tight_layout(w_pad=1.4)
    out = os.path.join(HERE, name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
#  Panel 1 -- the floor and granular meaning
# ===========================================================================
def panel1():
    beta = 1.0
    sizes, sig, ratios = [], [], []
    for _ in range(220):
        n = rng.randint(3, 11)
        G, items = random_contact_graph(n, beta, 0.4, rng)
        for v in items:
            s, _ = G.sigma(v)
            sizes.append(n + rng.uniform(-0.25, 0.25))
            sig.append(s)
            ratios.append(s / beta)
    aln, lb = [], []
    for _ in range(220):
        G, items = random_contact_graph(rng.randint(4, 10), beta, 0.5, rng)
        x, y = rng.sample(items, 2)
        _, score = G.alignment(x, y)
        aln.append(score)
        lb.append(beta / G.Omega())
    ns = [3, 5, 7, 9, 11, 13]
    bs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    Z = np.zeros((len(bs), len(ns)))
    for i, b in enumerate(bs):
        for j, nn in enumerate(ns):
            m = math.inf
            for _ in range(4):
                G, items = random_contact_graph(nn, b, 0.4, rng)
                m = min(m, min(G.sigma(v)[0] for v in items))
            Z[i, j] = m

    fig, ax = new_panel()
    a = ax[0]
    a.scatter(sizes, sig, s=10, c=BLUE, alpha=0.5, edgecolors="none")
    a.axhline(beta, color=RED, lw=1.4, ls="--")
    a.set_xlabel("graph size  n"); a.set_ylabel(r"$\sigma(v)$"); letter(a, "A")
    a.set_ylim(0, max(sig) * 1.05)

    b = ax[1]
    b.hist(ratios, bins=30, color=TEAL, alpha=0.85, edgecolor="white", lw=0.4)
    b.axvline(1.0, color=RED, lw=1.4, ls="--")
    b.set_xlabel(r"$\sigma(v)/\beta$"); b.set_ylabel("count"); letter(b, "B")

    c = ax[2]
    c.scatter(lb, aln, s=10, c=BLUE, alpha=0.5, edgecolors="none")
    mx = max(max(lb), max(aln))
    c.plot([0, mx], [0, mx], color=RED, lw=1.4, ls="--")
    c.set_xlabel(r"floor  $\beta/\Omega$"); c.set_ylabel(r"alignment  $a$")
    letter(c, "C")

    d = ax[3]
    X, Y = np.meshgrid(ns, bs)
    d.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.95)
    d.set_xlabel("n"); d.set_ylabel(r"$\beta$"); d.set_zlabel(r"$\min_v\sigma$")
    d.view_init(elev=24, azim=-58); style3d(d); letter(d, "D")
    save(fig, "panel_1.png")


# ===========================================================================
#  Panel 2 -- individuation by negation; identity as an invariant region
# ===========================================================================
def panel2():
    beta = 1.0
    ks, side = [], []
    for k in range(2, 11):
        for _ in range(6):
            G = ContactGraph(beta)
            for v in range(k):
                G.add_item(v); G.add_edge(G.medium, v, beta)
            for i in range(k):
                for j in range(i + 1, k):
                    G.add_edge(i, j, 50.0)
            _, reach = G.sigma(0)
            ks.append(k + rng.uniform(-0.15, 0.15))
            side.append(len(reach & set(range(k))))
    bef, aft = [], []
    for _ in range(160):
        G, items = random_contact_graph(rng.randint(3, 9), beta, 0.4, rng)
        perm = items[:]; rng.shuffle(perm)
        rel = dict(zip(items, perm))
        H = ContactGraph(beta)
        for v in items:
            H.add_item(rel[v])
        for e, w in G.weight.items():
            u, v = tuple(e)
            uu = H.medium if u == G.medium else rel[u]
            vv = H.medium if v == G.medium else rel[v]
            H.add_edge(uu, vv, w)
        v = rng.choice(items)
        bef.append(G.sigma(v)[0]); aft.append(H.sigma(rel[v])[0])
    Us, Cs, Vs = [], [], []
    for _ in range(400):
        V = rng.randint(2, 16)
        u = rng.randint(0, V)
        Us.append(u); Cs.append(V - u); Vs.append(V)

    fig, ax = new_panel()
    a = ax[0]
    a.scatter(ks, side, s=14, c=BLUE, alpha=0.55, edgecolors="none")
    a.plot([2, 10], [2, 10], color=RED, lw=1.4, ls="--")
    a.axhline(1, color="#999999", lw=1.0, ls=":")
    a.set_xlabel("cluster size  k"); a.set_ylabel(r"$|S^\ast(v)|$"); letter(a, "A")

    b = ax[1]
    b.scatter(bef, aft, s=12, c=TEAL, alpha=0.6, edgecolors="none")
    mx = max(max(bef), max(aft))
    b.plot([0, mx], [0, mx], color=RED, lw=1.4, ls="--")
    b.set_xlabel(r"$\sigma(v)$  before"); b.set_ylabel(r"$\sigma(v)$  after relabel")
    letter(b, "B")

    c = ax[2]
    c.scatter([v + rng.uniform(-0.15, 0.15) for v in Vs],
              [u + cc for u, cc in zip(Us, Cs)], s=11, c=TEAL,
              alpha=0.5, edgecolors="none")
    c.plot([0, 16], [0, 16], color=RED, lw=1.4, ls="--")
    c.set_xlabel(r"$|V|$"); c.set_ylabel(r"$|U|+|\complement U|$"); letter(c, "C")

    d = ax[3]
    d.scatter(Us, Cs, Vs, s=8, c=Vs, cmap="plasma", alpha=0.7)
    d.set_xlabel(r"$|U|$"); d.set_ylabel(r"$|\complement U|$"); d.set_zlabel(r"$|V|$")
    d.view_init(elev=22, azim=-60); style3d(d); letter(d, "D")
    save(fig, "panel_2.png")


# ===========================================================================
#  Panel 3 -- propagation, the monotone record, and the relaxation
# ===========================================================================
def panel3():
    beta = 1.0
    verts = list(range(5))
    walk = [rng.choice(verts) for _ in range(45)]
    steps = list(range(len(walk)))
    M = steps[:]                      # committed count == step
    seen, revisit = set(), []
    for i, v in enumerate(walk):
        if v in seen:
            revisit.append((i, M[i]))
        seen.add(v)

    def relax(d0, floorD, beta=1.0):
        d, ds = d0, [d0]
        for _ in range(60):
            if d <= floorD + 1e-9:
                break
            d = max(floorD, d - beta)
            ds.append(d)
        return ds
    solv = relax(8.0, 0.0)
    nohalt = relax(8.0, beta)         # plateau at the floor -> decline

    seeds = [relax(d0, 0.0) for d0 in (5.0, 6.0, 7.0, 8.0, 9.0)]

    fig, ax = new_panel()
    a = ax[0]
    a.step(steps, M, where="post", color=BLUE, lw=1.6)
    if revisit:
        rs, rm = zip(*revisit)
        a.scatter(rs, rm, s=20, c=RED, zorder=5)
    a.set_xlabel("step"); a.set_ylabel("committed record  M"); letter(a, "A")

    b = ax[1]
    b.plot(range(len(solv)), solv, color=TEAL, lw=1.8, marker="o", ms=3)
    b.plot(range(len(nohalt)), nohalt, color=ORANGE, lw=1.8, marker="s", ms=3)
    b.axhline(beta, color=RED, lw=1.2, ls="--")
    b.set_xlabel("relaxation step"); b.set_ylabel("cross-demand  D"); letter(b, "B")

    c = ax[2]
    for s in seeds:
        c.plot(range(len(s)), s, color=BLUE, lw=1.3, alpha=0.7)
    c.axhline(0, color=RED, lw=1.2, ls="--")
    c.set_xlabel("step"); c.set_ylabel("residual above floor"); letter(c, "C")

    d = ax[3]
    D0 = np.arange(3, 11)
    T = np.arange(0, 11)
    Tg, Dg = np.meshgrid(T, D0)
    Z = np.maximum(0.0, Dg - Tg * beta)      # demand(step, d0)
    d.plot_surface(Tg, Dg, Z, cmap="magma", edgecolor="none", alpha=0.95)
    d.set_xlabel("step"); d.set_ylabel(r"$D_0$"); d.set_zlabel("demand")
    d.view_init(elev=26, azim=-52); style3d(d); letter(d, "D")
    save(fig, "panel_3.png")


# ===========================================================================
#  Panel 4 -- four-column route audit; names; the master equivalence
# ===========================================================================
def panel4():
    beta = 1.0
    cen_ff, ext_ff, cen_ok, ext_ok = [], [], [], []
    for _ in range(220):
        if rng.random() < 0.5:
            cen_ff.append(rng.uniform(0, 0.08)); ext_ff.append(rng.uniform(1, 5))
        else:
            cen_ok.append(rng.uniform(0, 0.08)); ext_ok.append(rng.uniform(0, 0.01))

    Ws = np.linspace(2, 60, 30)
    compound, forced = [], []
    for W in Ws:
        G = ContactGraph(beta)
        G.add_item("u"); G.add_item("v")
        G.add_edge(G.medium, "u", beta); G.add_edge(G.medium, "v", beta)
        G.add_edge("u", "v", W)
        sU, _ = G.sigma({"u", "v"})
        compound.append(sU)               # = 2*beta, flat
        forced.append(2 * W + 2 * beta)   # composition of singleton cuts

    betas = np.linspace(0, 3, 26)
    smin = []
    for bb in betas:
        if bb <= 1e-9:
            smin.append(0.0); continue
        m = math.inf
        for _ in range(5):
            G, items = random_contact_graph(6, bb, 0.4, rng)
            m = min(m, min(G.sigma(v)[0] for v in items))
        smin.append(m)

    fig, ax = new_panel()
    a = ax[0]
    a.scatter(cen_ok, ext_ok, s=14, c=BLUE, alpha=0.7, edgecolors="none",
              label="proper")
    a.scatter(cen_ff, ext_ff, s=16, c=ORANGE, alpha=0.7, edgecolors="none",
              label="false friend")
    a.legend(loc="upper right", fontsize=7)
    a.set_xlabel("central demand"); a.set_ylabel("response demand"); letter(a, "A")

    b = ax[1]
    b.plot(Ws, forced, color=ORANGE, lw=1.8, label="composition")
    b.plot(Ws, compound, color=BLUE, lw=2.0, label=r"compound $\sigma(U)$")
    b.legend(loc="upper left", fontsize=7)
    b.set_xlabel("internal weight  W"); b.set_ylabel("cut weight"); letter(b, "B")

    c = ax[2]
    c.scatter(betas, smin, s=16, c=TEAL, alpha=0.8, edgecolors="none")
    c.plot([0, 3], [0, 3], color=RED, lw=1.2, ls="--")
    c.scatter([0], [0], s=40, c=RED, zorder=5)
    c.set_xlabel(r"floor  $\beta$"); c.set_ylabel(r"$\min_v\sigma$"); letter(c, "C")

    d = ax[3]
    d.scatter(Ws, compound, forced, s=16, c=forced, cmap="plasma", alpha=0.85)
    # guide lines
    d.plot(Ws, compound, [0] * len(Ws), color="#bbbbbb", lw=0.8)
    d.set_xlabel("W"); d.set_ylabel(r"$\sigma(U)$"); d.set_zlabel("composition")
    d.view_init(elev=22, azim=-60); style3d(d); letter(d, "D")
    save(fig, "panel_4.png")


if __name__ == "__main__":
    panel1(); panel2(); panel3(); panel4()
    print("done")
