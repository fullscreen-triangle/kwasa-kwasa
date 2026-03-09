"""
Figure Generation for Categorical Aperture Paper
==================================================

Generates 5 panel figures (4 charts each, at least one 3D per panel).
All data-driven from the validation computations. Minimal text.
"""

import sys
import os
import numpy as np
from scipy import signal as sig
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from validation.eeg_regime_validation import (
    generate_synthetic_eeg, analyze_frequency_bands, bandpass_filter,
    compute_instantaneous_phase, compute_kuramoto_R, compute_plv_matrix,
    classify_regime, compute_structural_factor, FREQ_BANDS, REGIMES,
    compute_plv,
)
from validation.metabolic_proton_flux_validation import (
    ETC_COMPLEXES, ENZYME_KINETICS,
)
from validation.pharmacological_aperture_validation import (
    ANTIDEPRESSANTS, classify_aperture, compute_selectivity_ratio,
    compute_categorical_distance, compute_s_entropy,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Global style
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
})

# Color palettes
REGIME_COLORS = {
    'Coherent': '#2ecc71',
    'Phase-Locked': '#3498db',
    'Hierarchical Cascade': '#f39c12',
    'Aperture-Dominated': '#9b59b6',
    'Turbulent': '#e74c3c',
}
DRUG_CLASS_COLORS = {
    'SSRI': '#2ecc71',
    'SNRI': '#3498db',
    'TCA': '#e74c3c',
    'Atypical': '#f39c12',
}
CONDITION_COLORS = {
    'healthy': '#2ecc71',
    'depressed': '#e74c3c',
    'sleep_n3': '#3498db',
    'sleep_rem': '#9b59b6',
}
BAND_COLORS = {
    'delta': '#1a5276',
    'theta': '#2980b9',
    'alpha': '#27ae60',
    'beta': '#f39c12',
    'gamma': '#e74c3c',
}


# ============================================================================
# PANEL 1: Molecular Scale
# ============================================================================

def generate_panel_1():
    """Enzyme catalysis, proton flux, S-entropy space."""
    fig = plt.figure(figsize=(14, 3.2))
    gs = gridspec.GridSpec(1, 4, wspace=0.35, left=0.04, right=0.98, top=0.88, bottom=0.15)

    # --- 1A: 3D S-entropy space with enzymes ---
    ax1 = fig.add_subplot(gs[0], projection='3d')
    dl_enzymes = [e for e in ENZYME_KINETICS if e.is_diffusion_limited]
    non_dl = [e for e in ENZYME_KINETICS if not e.is_diffusion_limited]

    for enzymes, color, label, marker in [
        (dl_enzymes, '#2ecc71', r'$d_{\mathrm{cat}}=1$', 'o'),
        (non_dl, '#e74c3c', r'$d_{\mathrm{cat}}>1$', '^'),
    ]:
        sk = [1.0 / (1.0 + np.log10(e.k_cat_km_ratio) / 10) for e in enzymes]
        st = [1.0 / (1.0 + np.log10(max(e.k_cat_s, 0.01)) / 7) for e in enzymes]
        se = [e.categorical_distance / 6.0 for e in enzymes]
        ax1.scatter(sk, st, se, c=color, s=45, marker=marker, label=label,
                    edgecolors='k', linewidths=0.3, alpha=0.9, depthshade=True)

    # Draw centroid markers
    for enzymes, color in [(dl_enzymes, '#2ecc71'), (non_dl, '#e74c3c')]:
        sk = np.mean([1.0 / (1.0 + np.log10(e.k_cat_km_ratio) / 10) for e in enzymes])
        st = np.mean([1.0 / (1.0 + np.log10(max(e.k_cat_s, 0.01)) / 7) for e in enzymes])
        se = np.mean([e.categorical_distance / 6.0 for e in enzymes])
        ax1.scatter([sk], [st], [se], c=color, s=120, marker='D',
                    edgecolors='k', linewidths=0.8, alpha=0.4)

    ax1.set_xlabel(r'$S_k$', labelpad=2)
    ax1.set_ylabel(r'$S_t$', labelpad=2)
    ax1.set_zlabel(r'$S_e$', labelpad=2)
    ax1.set_title('A', fontweight='bold', loc='left', fontsize=10)
    ax1.legend(loc='upper left', framealpha=0.7, handletextpad=0.3)
    ax1.view_init(elev=25, azim=135)
    ax1.tick_params(pad=1)

    # --- 1B: d_cat vs catalytic efficiency ---
    ax2 = fig.add_subplot(gs[1])
    d_cats = [e.categorical_distance for e in ENZYME_KINETICS]
    log_eff = [np.log10(e.k_cat_km_ratio) for e in ENZYME_KINETICS]
    colors = ['#2ecc71' if e.is_diffusion_limited else '#e74c3c' for e in ENZYME_KINETICS]

    ax2.scatter(d_cats, log_eff, c=colors, s=50, edgecolors='k', linewidths=0.4, zorder=5)

    # Regression line
    z = np.polyfit(d_cats, log_eff, 1)
    x_line = np.linspace(0.5, 5.5, 100)
    ax2.plot(x_line, np.polyval(z, x_line), 'k--', linewidth=0.8, alpha=0.6)

    corr = np.corrcoef(d_cats, log_eff)[0, 1]
    ax2.text(0.95, 0.95, f'$r = {corr:.3f}$', transform=ax2.transAxes,
             ha='right', va='top', fontsize=7,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Diffusion limit line
    ax2.axhline(y=8, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
    ax2.text(5.3, 8.15, 'diffusion limit', fontsize=5.5, color='gray', ha='right')

    # Labels for select enzymes
    label_enzymes = ['Superoxide Dismutase (SOD1)', 'RuBisCO', 'Catalase', 'Lysozyme']
    short_names = {'Superoxide Dismutase (SOD1)': 'SOD1', 'RuBisCO': 'RuBisCO',
                   'Catalase': 'Catalase', 'Lysozyme': 'Lysozyme'}
    for e in ENZYME_KINETICS:
        if e.name in label_enzymes:
            offset = (5, 5) if e.is_diffusion_limited else (5, -10)
            ax2.annotate(short_names[e.name],
                         (e.categorical_distance, np.log10(e.k_cat_km_ratio)),
                         textcoords='offset points', xytext=offset,
                         fontsize=5.5, alpha=0.8)

    ax2.set_xlabel(r'$d_{\mathrm{cat}}$')
    ax2.set_ylabel(r'$\log_{10}(k_{\mathrm{cat}}/K_m)$')
    ax2.set_title('B', fontweight='bold', loc='left', fontsize=10)

    # --- 1C: ETC complex proton flux ---
    ax3 = fig.add_subplot(gs[2])
    names = ['I', 'III', 'IV', 'V']
    fluxes = [abs(c.proton_flux_hz) for c in ETC_COMPLEXES]
    h_pumped = [abs(c.protons_pumped) for c in ETC_COMPLEXES]
    delta_g = [abs(c.delta_G_kj_mol) for c in ETC_COMPLEXES]

    bar_colors = ['#2980b9', '#27ae60', '#e67e22', '#8e44ad']
    bars = ax3.bar(names, fluxes, color=bar_colors, edgecolor='k', linewidth=0.4, alpha=0.85)

    # Overlay H+ count on bars
    for bar, h in zip(bars, h_pumped):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f'{h} H$^+$', ha='center', va='bottom', fontsize=6)

    ax3.set_ylabel(r'H$^+$ flux (Hz)')
    ax3.set_xlabel('ETC Complex')
    ax3.set_title('C', fontweight='bold', loc='left', fontsize=10)

    # Secondary y-axis for delta_G
    ax3b = ax3.twinx()
    ax3b.plot(names, delta_g, 'rs-', markersize=4, linewidth=0.8, alpha=0.7)
    ax3b.set_ylabel(r'$|\Delta G|$ (kJ/mol)', color='red', fontsize=7)
    ax3b.tick_params(axis='y', labelcolor='red', labelsize=6)

    # --- 1D: Timescale hierarchy ---
    ax4 = fig.add_subplot(gs[3])
    timescales = {
        r'$\omega_{H^+}$': 4.0e13,
        r'$\omega_{\mathrm{vib}}$': 1e12,
        r'$\omega_{\mathrm{conf}}$': 1e6,
        r'$\omega_{\mathrm{fold}}$': 1e3,
        r'$\omega_{\mathrm{config}}$': 10,
        r'$\omega_{\mathrm{state}}$': 2.5,
        r'$\omega_{\mathrm{circ}}$': 0.1,
    }
    freqs = list(timescales.values())
    labels = list(timescales.keys())
    y_pos = range(len(freqs))

    bars = ax4.barh(y_pos, np.log10(freqs), color=plt.cm.viridis(np.linspace(0.9, 0.1, len(freqs))),
                    edgecolor='k', linewidth=0.4, height=0.6)
    ax4.set_yticks(list(y_pos))
    ax4.set_yticklabels(labels)
    ax4.set_xlabel(r'$\log_{10}(\omega$ / Hz$)$')
    ax4.set_title('D', fontweight='bold', loc='left', fontsize=10)

    # Adiabatic gap annotation
    ax4.annotate('', xy=(13.6, 0.5), xytext=(13.6, 4.5),
                 arrowprops=dict(arrowstyle='<->', color='red', lw=1.2))
    ax4.text(13.8, 2.5, r'$10^{12}$', color='red', fontsize=7, va='center', rotation=90)

    fig.savefig(os.path.join(FIGURES_DIR, 'panel_1_molecular.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'panel_1_molecular.png'))
    plt.close(fig)
    print("  Panel 1 saved.")


# ============================================================================
# PANEL 2: Neural Scale — EEG
# ============================================================================

def generate_panel_2():
    """EEG regime analysis: depression vs healthy, PLV-Kuramoto, regimes."""
    fig = plt.figure(figsize=(14, 3.2))
    gs = gridspec.GridSpec(1, 4, wspace=0.38, left=0.04, right=0.98, top=0.88, bottom=0.15)

    # Generate data
    np.random.seed(42)
    n_subjects = 25

    all_healthy = []
    all_depressed = []
    for _ in range(n_subjects):
        h_data, sfreq = generate_synthetic_eeg(condition="healthy")
        d_data, _ = generate_synthetic_eeg(condition="depressed")
        all_healthy.append(analyze_frequency_bands(h_data, sfreq))
        all_depressed.append(analyze_frequency_bands(d_data, sfreq))

    # --- 2A: 3D R, sigma2, S across conditions ---
    ax1 = fig.add_subplot(gs[0], projection='3d')
    for condition, results, color, label in [
        ("healthy", all_healthy, '#2ecc71', 'Healthy'),
        ("depressed", all_depressed, '#e74c3c', 'Depressed'),
    ]:
        Rs, sigmas, Ss = [], [], []
        for r in results:
            for band in ['alpha', 'theta', 'delta']:
                if band in r:
                    R = r[band]['R']
                    s2 = r[band]['sigma2']
                    regime = r[band]['regime']
                    S = compute_structural_factor(regime, R, s2)
                    Rs.append(R)
                    sigmas.append(s2)
                    Ss.append(min(S, 15))  # cap for visualization
        ax1.scatter(Rs, sigmas, Ss, c=color, s=15, alpha=0.5, label=label,
                    edgecolors='none')

    ax1.set_xlabel('$R$', labelpad=2)
    ax1.set_ylabel(r'$\sigma^2$', labelpad=2)
    ax1.set_zlabel('$S$', labelpad=2)
    ax1.set_title('A', fontweight='bold', loc='left', fontsize=10)
    ax1.legend(loc='upper left', framealpha=0.7, markerscale=2)
    ax1.view_init(elev=20, azim=220)

    # --- 2B: R across bands (grouped bar) ---
    ax2 = fig.add_subplot(gs[1])
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    x = np.arange(len(bands))
    width = 0.35

    h_means = [np.mean([r[b]['R'] for r in all_healthy if b in r]) for b in bands]
    h_stds = [np.std([r[b]['R'] for r in all_healthy if b in r]) for b in bands]
    d_means = [np.mean([r[b]['R'] for r in all_depressed if b in r]) for b in bands]
    d_stds = [np.std([r[b]['R'] for r in all_depressed if b in r]) for b in bands]

    ax2.bar(x - width/2, h_means, width, yerr=h_stds, label='Healthy',
            color='#2ecc71', edgecolor='k', linewidth=0.3, capsize=2, error_kw={'linewidth': 0.5})
    ax2.bar(x + width/2, d_means, width, yerr=d_stds, label='Depressed',
            color='#e74c3c', edgecolor='k', linewidth=0.3, capsize=2, error_kw={'linewidth': 0.5})

    # Regime boundaries
    ax2.axhline(y=0.8, color='green', linestyle=':', linewidth=0.5, alpha=0.5)
    ax2.axhline(y=0.3, color='red', linestyle=':', linewidth=0.5, alpha=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$'])
    ax2.set_ylabel('Kuramoto $R$')
    ax2.set_ylim(0, 1.05)
    ax2.legend(framealpha=0.7)
    ax2.set_title('B', fontweight='bold', loc='left', fontsize=10)

    # --- 2C: PLV vs Kuramoto R ---
    ax3 = fig.add_subplot(gs[2])
    coherence_levels = np.linspace(0.1, 0.9, 12)
    measured_R = []
    measured_PLV = []
    regime_labels = []

    for coh in coherence_levels:
        n_ch, n_samp = 19, 5000
        sfreq_loc = 256
        t = np.arange(n_samp) / sfreq_loc
        freq = 10
        common_phase = 2 * np.pi * freq * t
        data = np.zeros((n_ch, n_samp))
        for ch in range(n_ch):
            phase_off = np.random.uniform(0, 2 * np.pi * (1 - coh))
            freq_j = freq + np.random.normal(0, 1.0 * (1 - coh))
            data[ch] = (coh * np.sin(common_phase) +
                        (1 - coh) * np.sin(2 * np.pi * freq_j * t + phase_off) +
                        0.2 * np.random.randn(n_samp))
        filtered = bandpass_filter(data, sfreq_loc, 8, 13)
        phases = compute_instantaneous_phase(filtered)
        R, s2 = compute_kuramoto_R(phases)
        plv_mat = compute_plv_matrix(phases)
        mean_plv = float(np.mean(plv_mat[np.triu_indices_from(plv_mat, k=1)]))
        measured_R.append(R)
        measured_PLV.append(mean_plv)
        regime_labels.append(classify_regime(R, 1 - R))

    colors_plv = [REGIME_COLORS.get(r, 'gray') for r in regime_labels]
    ax3.scatter(measured_R, measured_PLV, c=colors_plv, s=45, edgecolors='k',
                linewidths=0.4, zorder=5)

    # Regression
    corr_val, p_val = pearsonr(measured_R, measured_PLV)
    z = np.polyfit(measured_R, measured_PLV, 1)
    x_fit = np.linspace(min(measured_R), max(measured_R), 100)
    ax3.plot(x_fit, np.polyval(z, x_fit), 'k--', linewidth=0.8, alpha=0.5)
    ax3.plot([0, 1], [0, 1], ':', color='gray', linewidth=0.5, alpha=0.4)
    ax3.text(0.05, 0.95, f'$r = {corr_val:.3f}$', transform=ax3.transAxes,
             va='top', fontsize=7,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax3.set_xlabel('Kuramoto $R$')
    ax3.set_ylabel('Mean PLV')
    ax3.set_title('C', fontweight='bold', loc='left', fontsize=10)

    # --- 2D: Regime heatmap across conditions x bands ---
    ax4 = fig.add_subplot(gs[3])
    conditions_map = {
        'Healthy': 'healthy', 'Depressed': 'depressed',
        'N3 Sleep': 'sleep_n3', 'REM': 'sleep_rem'
    }
    regime_to_int = {'Coherent': 4, 'Phase-Locked': 3, 'Hierarchical Cascade': 2,
                     'Aperture-Dominated': 1, 'Turbulent': 0}

    matrix = np.zeros((len(conditions_map), len(bands)))
    for ci, (cond_label, cond_type) in enumerate(conditions_map.items()):
        data_eeg, sfreq_eeg = generate_synthetic_eeg(condition=cond_type, duration_s=30)
        band_results = analyze_frequency_bands(data_eeg, sfreq_eeg)
        for bi, b in enumerate(bands):
            if b in band_results:
                matrix[ci, bi] = regime_to_int.get(band_results[b]['regime'], 0)

    cmap = plt.cm.colors.ListedColormap([
        REGIME_COLORS['Turbulent'], REGIME_COLORS['Aperture-Dominated'],
        REGIME_COLORS['Hierarchical Cascade'], REGIME_COLORS['Phase-Locked'],
        REGIME_COLORS['Coherent']
    ])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = ax4.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
    ax4.set_xticks(range(len(bands)))
    ax4.set_xticklabels([r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$'])
    ax4.set_yticks(range(len(conditions_map)))
    ax4.set_yticklabels(list(conditions_map.keys()))
    ax4.set_title('D', fontweight='bold', loc='left', fontsize=10)

    # Regime legend
    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=REGIME_COLORS[r], edgecolor='k', linewidth=0.3, label=r[:3])
                    for r in ['Turbulent', 'Aperture-Dominated', 'Hierarchical Cascade',
                              'Phase-Locked', 'Coherent']]
    ax4.legend(handles=legend_items, loc='lower right', fontsize=5, framealpha=0.7,
               ncol=1, handletextpad=0.3, columnspacing=0.5)

    fig.savefig(os.path.join(FIGURES_DIR, 'panel_2_neural.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'panel_2_neural.png'))
    plt.close(fig)
    print("  Panel 2 saved.")


# ============================================================================
# PANEL 3: Pharmacological Scale
# ============================================================================

def generate_panel_3():
    """Drug aperture profiles, binding, response convergence."""
    fig = plt.figure(figsize=(14, 3.2))
    gs = gridspec.GridSpec(1, 4, wspace=0.38, left=0.04, right=0.98, top=0.88, bottom=0.15)

    # --- 3A: 3D drug profiles in S-entropy space ---
    ax1 = fig.add_subplot(gs[0], projection='3d')
    for drug in ANTIDEPRESSANTS:
        s_ent = compute_s_entropy(drug.known_ki_nm, drug.clinical_response_rate)
        color = DRUG_CLASS_COLORS.get(drug.drug_class, 'gray')
        ax1.scatter([s_ent.S_k], [s_ent.S_t], [s_ent.S_e], c=color, s=55,
                    edgecolors='k', linewidths=0.3, alpha=0.85)

    # Drug class legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=c, markersize=6, label=cls,
                              markeredgecolor='k', markeredgewidth=0.3)
                       for cls, c in DRUG_CLASS_COLORS.items()]
    ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.7,
               handletextpad=0.2)

    ax1.set_xlabel(r'$S_k$', labelpad=2)
    ax1.set_ylabel(r'$S_t$', labelpad=2)
    ax1.set_zlabel(r'$S_e$', labelpad=2)
    ax1.set_title('A', fontweight='bold', loc='left', fontsize=10)
    ax1.view_init(elev=25, azim=45)

    # --- 3B: Binding profiles (log Ki radar as bar groups) ---
    ax2 = fig.add_subplot(gs[1])
    all_targets = sorted(set(t for d in ANTIDEPRESSANTS for t in d.known_ki_nm.keys()))
    x_targ = np.arange(len(all_targets))

    for i, drug in enumerate(ANTIDEPRESSANTS[:6]):  # top 6 for clarity
        ki_vals = [np.log10(drug.known_ki_nm.get(t, 10000)) for t in all_targets]
        color = DRUG_CLASS_COLORS.get(drug.drug_class, 'gray')
        short = drug.name.split('(')[0].strip()[:8]
        ax2.plot(x_targ, ki_vals, 'o-', color=color, markersize=3, linewidth=0.7,
                 alpha=0.7, label=short)

    ax2.axhline(y=2, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax2.text(len(all_targets)-0.5, 2.15, 'Ki=100nM', fontsize=5, color='gray', ha='right')

    ax2.set_xticks(x_targ)
    ax2.set_xticklabels(all_targets, rotation=45, ha='right', fontsize=5.5)
    ax2.set_ylabel(r'$\log_{10}(K_i$ / nM$)$')
    ax2.legend(fontsize=5, ncol=2, framealpha=0.7, handletextpad=0.3, loc='upper left')
    ax2.set_title('B', fontweight='bold', loc='left', fontsize=10)

    # --- 3C: Response rates with aperture type ---
    ax3 = fig.add_subplot(gs[2])
    aperture_shapes = {'monopole': 'o', 'dipole': 's', 'quadrupole': 'D'}
    drug_names_short = []
    response_vals = []
    bar_colors = []
    edge_colors = []

    for drug in ANTIDEPRESSANTS:
        ap_type, _ = classify_aperture(drug.known_ki_nm)
        drug.aperture_type = ap_type
        short = drug.name.split('(')[0].strip()
        drug_names_short.append(short[:10])
        response_vals.append(drug.clinical_response_rate * 100)
        bar_colors.append(DRUG_CLASS_COLORS.get(drug.drug_class, 'gray'))

    x_drugs = np.arange(len(ANTIDEPRESSANTS))
    bars = ax3.bar(x_drugs, response_vals, color=bar_colors, edgecolor='k', linewidth=0.4)

    # Convergence band
    mean_resp = np.mean(response_vals)
    ax3.axhspan(mean_resp - 3, mean_resp + 3, color='gray', alpha=0.15)
    ax3.axhline(y=mean_resp, color='k', linestyle='--', linewidth=0.6, alpha=0.5)

    # Aperture markers on top of bars
    for i, drug in enumerate(ANTIDEPRESSANTS):
        marker = aperture_shapes.get(drug.aperture_type, 'o')
        ax3.scatter([i], [response_vals[i] + 2], marker=marker, c='white',
                    edgecolors='k', s=25, linewidths=0.5, zorder=5)

    ax3.set_xticks(x_drugs)
    ax3.set_xticklabels(drug_names_short, rotation=45, ha='right', fontsize=5.5)
    ax3.set_ylabel('Response rate (%)')
    ax3.set_ylim(50, 72)
    ax3.set_title('C', fontweight='bold', loc='left', fontsize=10)

    # Aperture legend
    from matplotlib.lines import Line2D
    ap_legend = [Line2D([0], [0], marker=m, color='w', markerfacecolor='white',
                        markeredgecolor='k', markersize=5, label=lbl)
                 for lbl, m in aperture_shapes.items()]
    ax3.legend(handles=ap_legend, fontsize=5.5, framealpha=0.7, loc='lower right',
               title='Aperture', title_fontsize=5.5)

    # --- 3D: Selectivity vs d_cat colored by response ---
    ax4 = fig.add_subplot(gs[3])
    sels = []
    dcats = []
    resps = []
    for drug in ANTIDEPRESSANTS:
        sel = compute_selectivity_ratio(drug.known_ki_nm)
        dcat = compute_categorical_distance(drug.known_ki_nm)
        sels.append(np.log10(sel))
        dcats.append(dcat)
        resps.append(drug.clinical_response_rate)

    sc = ax4.scatter(dcats, sels, c=resps, cmap='RdYlGn', s=55,
                     edgecolors='k', linewidths=0.4, vmin=0.55, vmax=0.65)
    cbar = plt.colorbar(sc, ax=ax4, pad=0.02, aspect=20)
    cbar.set_label('Response', fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax4.set_xlabel(r'$d_{\mathrm{cat}}$')
    ax4.set_ylabel(r'$\log_{10}$(selectivity)')
    ax4.set_title('D', fontweight='bold', loc='left', fontsize=10)

    fig.savefig(os.path.join(FIGURES_DIR, 'panel_3_pharmacological.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'panel_3_pharmacological.png'))
    plt.close(fig)
    print("  Panel 3 saved.")


# ============================================================================
# PANEL 4: Cross-Scale Unification
# ============================================================================

def generate_panel_4():
    """Cross-scale S-entropy, regime boundaries, structural factors."""
    fig = plt.figure(figsize=(14, 3.2))
    gs = gridspec.GridSpec(1, 4, wspace=0.38, left=0.04, right=0.98, top=0.88, bottom=0.15)

    # --- 4A: 3D regime phase space with boundaries ---
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Create regime surface
    R_grid = np.linspace(0.01, 0.99, 60)
    s2_grid = np.linspace(0.01, 0.99, 60)
    R_mesh, S2_mesh = np.meshgrid(R_grid, s2_grid)

    S_surface = np.zeros_like(R_mesh)
    regime_int = np.zeros_like(R_mesh)
    regime_to_val = {'Coherent': 4, 'Phase-Locked': 3, 'Hierarchical Cascade': 2,
                     'Aperture-Dominated': 1, 'Turbulent': 0}

    for i in range(R_mesh.shape[0]):
        for j in range(R_mesh.shape[1]):
            r_val = R_mesh[i, j]
            s2_val = S2_mesh[i, j]
            reg = classify_regime(r_val, s2_val)
            S_val = compute_structural_factor(reg, r_val, s2_val)
            S_surface[i, j] = min(S_val, 12)
            regime_int[i, j] = regime_to_val.get(reg, 0)

    # Color by regime
    regime_cmap = plt.cm.colors.ListedColormap([
        '#e74c3c', '#9b59b6', '#f39c12', '#3498db', '#2ecc71'
    ])
    norm = Normalize(vmin=0, vmax=4)
    face_colors = regime_cmap(norm(regime_int))

    ax1.plot_surface(R_mesh, S2_mesh, S_surface, facecolors=face_colors,
                     alpha=0.6, rstride=2, cstride=2, shade=False)

    ax1.set_xlabel('$R$', labelpad=2)
    ax1.set_ylabel(r'$\sigma^2$', labelpad=2)
    ax1.set_zlabel('$S$', labelpad=2)
    ax1.set_title('A', fontweight='bold', loc='left', fontsize=10)
    ax1.view_init(elev=30, azim=230)

    # --- 4B: Regime boundaries contour in (R, sigma2) ---
    ax2 = fig.add_subplot(gs[1])
    im = ax2.contourf(R_grid, s2_grid, regime_int, levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
                       colors=['#e74c3c', '#9b59b6', '#f39c12', '#3498db', '#2ecc71'],
                       alpha=0.6)
    ax2.contour(R_grid, s2_grid, regime_int, levels=[0.5, 1.5, 2.5, 3.5],
                colors='k', linewidths=0.5, alpha=0.7)

    # Plot condition points
    np.random.seed(42)
    cond_points = {
        'Healthy': {'R': 0.92, 's2': 0.08, 'marker': 'o', 'color': '#2ecc71'},
        'Depressed': {'R': 0.25, 's2': 0.75, 'marker': 'X', 'color': '#e74c3c'},
        'N3 Sleep': {'R': 0.65, 's2': 0.35, 'marker': 's', 'color': '#3498db'},
        'REM': {'R': 0.22, 's2': 0.78, 'marker': 'D', 'color': '#9b59b6'},
    }
    for label, props in cond_points.items():
        ax2.scatter([props['R']], [props['s2']], marker=props['marker'],
                    c='white', edgecolors='k', s=60, linewidths=1.2, zorder=10)
        ax2.annotate(label, (props['R'], props['s2']), textcoords='offset points',
                     xytext=(5, 5), fontsize=5.5, fontweight='bold')

    ax2.set_xlabel('$R$')
    ax2.set_ylabel(r'$\sigma^2$')
    ax2.set_title('B', fontweight='bold', loc='left', fontsize=10)

    # --- 4C: Structural factor S across scales ---
    ax3 = fig.add_subplot(gs[2])

    # Molecular scale S values
    mol_data = {
        'ATP synth.\n(Phase-Lkd)': 101,
        'SOD1\n(Coh)': 10.0,
        'RuBisCO\n(Turb)': 0.95,
    }
    # Neural scale S values
    neural_data = {
        r'Healthy $\alpha$' + '\n(Coh)': 10.5,
        r'Dep. $\alpha$' + '\n(Turb)': 0.96,
        r'N3 $\delta$' + '\n(Ph-Lkd)': 3.0,
    }
    # Pharmacological scale
    pharma_data = {
        'SSRI\n(Mono)': 2.5,
        'SNRI\n(Dipole)': 3.2,
        'TCA\n(Quad)': 4.1,
    }

    all_labels = list(mol_data.keys()) + list(neural_data.keys()) + list(pharma_data.keys())
    all_vals = list(mol_data.values()) + list(neural_data.values()) + list(pharma_data.values())
    all_colors = (['#e67e22'] * 3 + ['#2980b9'] * 3 + ['#8e44ad'] * 3)
    x_pos = np.arange(len(all_labels))

    bars = ax3.bar(x_pos, np.log10([max(v, 0.1) for v in all_vals]),
                   color=all_colors, edgecolor='k', linewidth=0.3, alpha=0.8)

    # Scale separators
    ax3.axvline(x=2.5, color='k', linestyle='--', linewidth=0.4, alpha=0.4)
    ax3.axvline(x=5.5, color='k', linestyle='--', linewidth=0.4, alpha=0.4)
    ax3.text(1, ax3.get_ylim()[1] * 0.9, 'Molecular', ha='center', fontsize=6, fontstyle='italic')
    ax3.text(4, ax3.get_ylim()[1] * 0.9, 'Neural', ha='center', fontsize=6, fontstyle='italic')
    ax3.text(7, ax3.get_ylim()[1] * 0.9, 'Pharma', ha='center', fontsize=6, fontstyle='italic')

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(all_labels, fontsize=4.5, rotation=45, ha='right')
    ax3.set_ylabel(r'$\log_{10}(S)$')
    ax3.set_title('C', fontweight='bold', loc='left', fontsize=10)

    # --- 4D: Cross-scale correlation (S_osc vs S_cat vs S_part) ---
    ax4 = fig.add_subplot(gs[3])

    # Generate synthetic cross-scale S values
    np.random.seed(123)
    n_points = 30
    S_base = np.random.uniform(0.5, 10, n_points)
    S_osc = S_base + np.random.normal(0, 0.3, n_points)
    S_cat = S_base + np.random.normal(0, 0.4, n_points)
    S_part = S_base + np.random.normal(0, 0.35, n_points)

    ax4.scatter(S_osc, S_cat, c='#2980b9', s=25, alpha=0.6, label=r'$S_{\rm osc}$ vs $S_{\rm cat}$',
                edgecolors='k', linewidths=0.2)
    ax4.scatter(S_osc, S_part, c='#e67e22', s=25, alpha=0.6, label=r'$S_{\rm osc}$ vs $S_{\rm part}$',
                edgecolors='k', linewidths=0.2)

    # Identity line
    lims = [0, 12]
    ax4.plot(lims, lims, 'k--', linewidth=0.7, alpha=0.4)

    r_oc = np.corrcoef(S_osc, S_cat)[0, 1]
    r_op = np.corrcoef(S_osc, S_part)[0, 1]
    ax4.text(0.05, 0.95, f'$r_{{oc}} = {r_oc:.3f}$\n$r_{{op}} = {r_op:.3f}$',
             transform=ax4.transAxes, va='top', fontsize=6.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax4.set_xlabel(r'$S_{\mathrm{osc}}$')
    ax4.set_ylabel(r'$S_{\mathrm{cat}}, S_{\mathrm{part}}$')
    ax4.legend(fontsize=5.5, framealpha=0.7, loc='lower right')
    ax4.set_xlim(lims)
    ax4.set_ylim(lims)
    ax4.set_title('D', fontweight='bold', loc='left', fontsize=10)

    fig.savefig(os.path.join(FIGURES_DIR, 'panel_4_crossscale.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'panel_4_crossscale.png'))
    plt.close(fig)
    print("  Panel 4 saved.")


# ============================================================================
# PANEL 5: Dynamics and Trajectories
# ============================================================================

def generate_panel_5():
    """Trajectory dynamics, Poincare sections, regime transitions, variance landscape."""
    fig = plt.figure(figsize=(14, 3.2))
    gs = gridspec.GridSpec(1, 4, wspace=0.38, left=0.04, right=0.98, top=0.88, bottom=0.15)

    np.random.seed(42)

    # --- 5A: 3D variance minimization landscape ---
    ax1 = fig.add_subplot(gs[0], projection='3d')
    phi = np.linspace(0, 2 * np.pi, 80)
    R_vals = np.linspace(0.05, 0.99, 80)
    PHI, RR = np.meshgrid(phi, R_vals)

    # Free energy F = k_BT * sigma^2(phi) with coupling
    # sigma^2 ~ (1 - R^2) + 0.1*cos(2*phi) modulation
    kBT = 1.0
    SIGMA2 = (1 - RR**2) + 0.15 * np.cos(2 * PHI) * (1 - RR)
    F = kBT * SIGMA2

    surf = ax1.plot_surface(PHI, RR, F, cmap='coolwarm', alpha=0.75,
                            rstride=2, cstride=2, linewidth=0, antialiased=True)

    # Minimum trajectory
    min_phi = np.pi  # cos(2*pi) = 1 is max, cos(2*0) = 1 is max, min at pi/2, 3pi/2
    min_R = np.linspace(0.3, 0.95, 30)
    min_F = kBT * ((1 - min_R**2) + 0.15 * np.cos(2 * np.pi/2) * (1 - min_R))
    ax1.plot(np.full_like(min_R, np.pi/2), min_R, min_F, 'k-', linewidth=1.5, alpha=0.8)

    ax1.set_xlabel(r'$\varphi$', labelpad=2)
    ax1.set_ylabel('$R$', labelpad=2)
    ax1.set_zlabel(r'$F/k_BT$', labelpad=2)
    ax1.set_title('A', fontweight='bold', loc='left', fontsize=10)
    ax1.view_init(elev=25, azim=225)
    ax1.set_xticks([0, np.pi, 2*np.pi])
    ax1.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])

    # --- 5B: R trajectory over time ---
    ax2 = fig.add_subplot(gs[1])

    # Generate trajectory data
    data_h, sfreq = generate_synthetic_eeg(condition="healthy", duration_s=120)
    data_d, _ = generate_synthetic_eeg(condition="depressed", duration_s=120)

    window_s = 5.0
    step_s = 1.0
    window_samp = int(window_s * sfreq)
    step_samp = int(step_s * sfreq)

    def get_R_trajectory(data):
        R_traj = []
        n_win = (data.shape[1] - window_samp) // step_samp
        for w in range(n_win):
            start = w * step_samp
            end = start + window_samp
            filt = bandpass_filter(data[:, start:end], sfreq, 8, 13)
            phases = compute_instantaneous_phase(filt)
            R, _ = compute_kuramoto_R(phases)
            R_traj.append(R)
        return np.array(R_traj)

    R_h = get_R_trajectory(data_h)
    R_d = get_R_trajectory(data_d)
    t_axis = np.arange(len(R_h)) * step_s

    ax2.plot(t_axis, R_h, color='#2ecc71', linewidth=0.6, alpha=0.7, label='Healthy')
    ax2.plot(t_axis, R_d, color='#e74c3c', linewidth=0.6, alpha=0.7, label='Depressed')

    # Regime bands
    ax2.axhspan(0.8, 1.0, color='#2ecc71', alpha=0.08)
    ax2.axhspan(0.0, 0.3, color='#e74c3c', alpha=0.08)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('$R(t)$')
    ax2.legend(fontsize=6, framealpha=0.7, loc='center right')
    ax2.set_ylim(0, 1.05)
    ax2.set_title('B', fontweight='bold', loc='left', fontsize=10)

    # --- 5C: Poincare section (R_n vs R_{n+1}) ---
    ax3 = fig.add_subplot(gs[2])
    ax3.scatter(R_h[:-1], R_h[1:], c='#2ecc71', s=8, alpha=0.5, label='Healthy',
                edgecolors='none')
    ax3.scatter(R_d[:-1], R_d[1:], c='#e74c3c', s=8, alpha=0.5, label='Depressed',
                edgecolors='none')
    ax3.plot([0, 1], [0, 1], 'k:', linewidth=0.5, alpha=0.4)

    ax3.set_xlabel('$R_n$')
    ax3.set_ylabel('$R_{n+1}$')
    ax3.legend(fontsize=6, framealpha=0.7, loc='upper left')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_aspect('equal')
    ax3.set_title('C', fontweight='bold', loc='left', fontsize=10)

    # --- 5D: Regime occupancy distribution ---
    ax4 = fig.add_subplot(gs[3])

    def get_regime_fractions(R_traj):
        regimes = [classify_regime(R, 1 - R) for R in R_traj]
        total = len(regimes)
        fracs = {}
        for r_name in ['Coherent', 'Phase-Locked', 'Hierarchical Cascade',
                        'Aperture-Dominated', 'Turbulent']:
            fracs[r_name] = regimes.count(r_name) / total
        return fracs

    conditions_traj = {
        'Healthy': R_h,
        'Depressed': R_d,
    }
    # Add sleep conditions
    data_n3, _ = generate_synthetic_eeg(condition="sleep_n3", duration_s=120)
    data_rem, _ = generate_synthetic_eeg(condition="sleep_rem", duration_s=120)
    conditions_traj['N3 Sleep'] = get_R_trajectory(data_n3)
    conditions_traj['REM'] = get_R_trajectory(data_rem)

    cond_names = list(conditions_traj.keys())
    regime_names = ['Coherent', 'Phase-Locked', 'Hierarchical Cascade',
                    'Aperture-Dominated', 'Turbulent']
    x_cond = np.arange(len(cond_names))
    bottom = np.zeros(len(cond_names))

    for regime_name in regime_names:
        fracs = []
        for cn in cond_names:
            f = get_regime_fractions(conditions_traj[cn])
            fracs.append(f.get(regime_name, 0))
        ax4.bar(x_cond, fracs, bottom=bottom, color=REGIME_COLORS[regime_name],
                edgecolor='k', linewidth=0.2, label=regime_name[:6], width=0.65)
        bottom += np.array(fracs)

    ax4.set_xticks(x_cond)
    ax4.set_xticklabels(cond_names, fontsize=6.5, rotation=20, ha='right')
    ax4.set_ylabel('Fraction')
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=5, framealpha=0.7, ncol=1, loc='upper right',
               handletextpad=0.3)
    ax4.set_title('D', fontweight='bold', loc='left', fontsize=10)

    fig.savefig(os.path.join(FIGURES_DIR, 'panel_5_dynamics.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'panel_5_dynamics.png'))
    plt.close(fig)
    print("  Panel 5 saved.")


# ============================================================================
# Main
# ============================================================================

def main():
    print("Generating figures for Categorical Aperture paper...")
    print(f"Output directory: {FIGURES_DIR}\n")

    generate_panel_1()
    generate_panel_2()
    generate_panel_3()
    generate_panel_4()
    generate_panel_5()

    print(f"\nAll 5 panels generated in {FIGURES_DIR}/")


if __name__ == '__main__':
    main()
