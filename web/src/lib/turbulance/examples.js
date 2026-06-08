/* Tutorial programs for the kwasa-kwasa playground.
 * Each is a small, runnable Turbulance script; together they double as the
 * acceptance tests the Rust implementation must also satisfy. */

export const examples = [
  {
    id: "basics",
    title: "1 · Bindings and functions",
    description:
      "item binds a value; funxn declares a function (note funxn, not function). " +
      "print writes to the console; {} is a placeholder.",
    code: `funxn greet(name):
    item message = "Hello, " + name
    print(message)
    return message

funxn main():
    item who = "kwasa-kwasa"
    greet(who)
    print("2 + 2 * 3 = {}", 2 + 2 * 3)
`,
  },
  {
    id: "control",
    title: "2 · given, within, considering",
    description:
      "given is the conditional (not if). considering iterates over a collection. " +
      "within opens a scope for guarded actions.",
    code: `funxn main():
    item scores = [0.9, 0.4, 0.7, 0.2, 0.85]
    item high = 0

    considering score in scores:
        given score > 0.5:
            high = high + 1

    print("scores above 0.5: {}", high)

    within scores:
        given len(scores) > 3:
            print("a sizeable batch of {} scores", len(scores))
`,
  },
  {
    id: "proposition",
    title: "3 · Propositions and graded support",
    description:
      "A proposition is a hypothesis decomposed into motions. support and " +
      "contradict accumulate graded evidence; each motion gets a confidence and " +
      "a verdict (Supported / Contradicted / Inconclusive).",
    code: `funxn main():
    item missing_count = 0
    item outlier_rate = 0.05

    proposition DataQuality:
        motion CompleteData("Dataset has no missing values")
        motion ReasonableValues("Values lie within expected ranges")

        within "dataset":
            given missing_count == 0:
                support CompleteData with_confidence(0.95)
            given outlier_rate < 0.1:
                support ReasonableValues with_confidence(0.9)
            given outlier_rate > 0.2:
                contradict ReasonableValues with_confidence(0.6)

    print("Verdict: {} (score {})", DataQuality.verdict, DataQuality.score)
`,
  },
  {
    id: "points",
    title: "4 · Points and resolution",
    description:
      "A point is a datum that carries its own confidence. resolve collapses a " +
      "point to a determinate entity under the maximum-likelihood strategy.",
    code: `funxn main():
    point diagnosis = {
        content: "elevated marker, ambiguous aetiology",
        confidence: 0.7
    }

    item determined = resolve diagnosis
    print("resolved: {}", determined.value)
    print("confidence: {}", determined.confidence)
`,
  },
  {
    id: "confidence",
    title: "5 · The confidence algebra",
    description:
      "Two motions, several pieces of corroborating and opposing evidence. " +
      "Watch how graded support aggregates and how a contention pulls a score down.",
    code: `funxn main():
    proposition DrugEfficacy:
        motion ReducesSymptoms("The drug reduces symptom severity")
        motion MinimalSideEffects("Side effects are minimal")

        given true:
            support ReducesSymptoms with_confidence(0.8)
            support ReducesSymptoms with_confidence(0.7)
            support MinimalSideEffects with_confidence(0.9)
            contradict MinimalSideEffects with_confidence(0.5)

    print("Overall verdict: {}", DrugEfficacy.verdict)
`,
  },
  {
    id: "massspec",
    title: "6 · Polyglot: delegate to Python",
    description:
      "kwasa-kwasa is not a replacement for other languages — it orchestrates them. " +
      "Here Turbulance hands the numeric work to a Python specialist (numpy/scipy, " +
      "running in your browser via Pyodide), then EVALUATES the result against a " +
      "hypothesis. Open lavoisier.py to see the specialist. First run loads Python (~7 MB).",
    code: `// Mass-spec biomarker discovery.
// Turbulance is the conductor and the judge; Python does the number-crunching.

hypothesis BiomarkerDiscovery:
    claim: "Metabolomic peaks separate diabetic from control samples"
    success_criteria:
        - separation: 0.6
        - num_features: 3
        - quality: 0.7

funxn main():
    print("Delegating spectrum analysis to a Python specialist (numpy/scipy)...")

    // Hand the heavy numeric work to lavoisier.py (runs in-browser via Pyodide).
    // Contract: JSON in -> the specialist's analyze_spectrum(data) -> JSON out.
    item analysis = trebuchet.delegate("lavoisier.py", "analyze_spectrum", {
        n_samples: 40,
        noise: 0.15,
        seed: 7
    })

    print("Python found {} peaks, {} candidate features", analysis.num_peaks, analysis.num_features)
    print("Group separation: {}  |  spectrum quality: {}", analysis.separation, analysis.quality)

    // Now Turbulance evaluates the Python output against the hypothesis.
    proposition BiomarkerEvidence:
        motion Separates("Peaks separate the two groups")
        motion EnoughFeatures("Enough candidate biomarkers were found")
        motion GoodQuality("Spectrum quality is acceptable")

        given analysis.separation >= BiomarkerDiscovery.success_criteria.separation:
            support Separates with_confidence(analysis.separation)
        given analysis.num_features >= BiomarkerDiscovery.success_criteria.num_features:
            support EnoughFeatures with_confidence(0.9)
        given analysis.quality >= BiomarkerDiscovery.success_criteria.quality:
            support GoodQuality with_confidence(analysis.quality)
        given analysis.quality < BiomarkerDiscovery.success_criteria.quality:
            contradict GoodQuality with_confidence(0.6)

    print("Verdict: {}", BiomarkerEvidence.verdict)
    return BiomarkerEvidence
`,
    files: {
      "lavoisier.py": `# A self-contained mass-spec "specialist" called by Turbulance via trebuchet.delegate.
# In the real framework this would be the full Lavoisier pipeline; here it does the
# same SHAPE of work (synthesize -> peak-pick -> group statistics) with numpy/scipy.
import numpy as np
from scipy import signal


def analyze_spectrum(params):
    p = params if isinstance(params, dict) else {}
    n = int(p.get("n_samples", 40))
    noise = float(p.get("noise", 0.15))
    seed = int(p.get("seed", 7))
    rng = np.random.default_rng(seed)

    mz = np.linspace(50, 500, 1000)
    peaks = [(120, 1.0), (180, 0.8), (250, 0.6), (340, 0.9)]

    def spectrum(diabetic):
        base = np.zeros_like(mz)
        for center, amp in peaks:
            # two peaks are up-regulated in the diabetic group
            a = amp * (1.6 if diabetic and center in (180, 340) else 1.0)
            base += a * np.exp(-0.5 * ((mz - center) / 3.0) ** 2)
        base += noise * rng.standard_normal(mz.shape)
        return base

    half = n // 2
    diabetic = np.array([spectrum(True) for _ in range(half)])
    control = np.array([spectrum(False) for _ in range(n - half)])

    mean_all = np.mean(np.vstack([diabetic, control]), axis=0)
    peak_idx, _ = signal.find_peaks(mean_all, height=0.3, distance=10)

    feats = []
    for idx in peak_idx:
        d = diabetic[:, idx]
        c = control[:, idx]
        diff = abs(float(np.mean(d) - np.mean(c)))
        pooled = float(np.sqrt((np.var(d) + np.var(c)) / 2) + 1e-9)
        feats.append({"mz": float(mz[idx]), "effect_size": diff / pooled})

    feats.sort(key=lambda f: f["effect_size"], reverse=True)
    significant = [f for f in feats if f["effect_size"] > 1.0]

    top = feats[:5]
    mean_effect = float(np.mean([f["effect_size"] for f in top])) if top else 0.0
    separation = float(1 - np.exp(-mean_effect / 2.0))   # squash to [0, 1]

    snr = float(np.max(mean_all) / (noise + 1e-9))
    quality = float(min(1.0, snr / 8.0))

    return {
        "num_peaks": int(len(peak_idx)),
        "num_features": int(len(significant)),
        "separation": round(separation, 3),
        "quality": round(quality, 3),
        "top_features": [
            {"mz": round(f["mz"], 1), "effect_size": round(f["effect_size"], 2)} for f in top
        ],
    }
`,
    },
  },
];

export const defaultExample = examples[0];
