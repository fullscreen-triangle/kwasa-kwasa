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
    id: "experiment",
    title: "6 · Application: verify an experiment design",
    description:
      "A program that does not compute a result but CHECKS the logic of a plan. " +
      "The experiment design is encoded as motions; given-guards verify power, " +
      "controls, randomization and blinding, and the proposition returns a verdict. " +
      "Pure Turbulance — no Python, no AI.",
    code: `// Verify whether a proposed experiment design is sound enough to run.
// Turbulance checks the *logic* of the plan, not any data.

hypothesis CleanTrial:
    claim: "The design is sound enough to run"
    success_criteria:
        - min_sample: 30
        - min_power: 0.8

funxn check_design(n_per_group, has_control, randomized, blinded, power):
    proposition DesignIsSound:
        motion Powered("Sample size and power are adequate")
        motion Controlled("There is a control group")
        motion Randomized("Assignment is randomized")
        motion Blinded("The trial is blinded")

        given n_per_group >= CleanTrial.success_criteria.min_sample and power >= CleanTrial.success_criteria.min_power:
            support Powered with_confidence(0.9)
        given n_per_group < CleanTrial.success_criteria.min_sample or power < CleanTrial.success_criteria.min_power:
            contradict Powered with_confidence(0.8)
        given has_control:
            support Controlled with_confidence(0.95)
        given randomized:
            support Randomized with_confidence(0.9)
        given blinded:
            support Blinded with_confidence(0.85)

    return DesignIsSound

funxn main():
    print("Checking a well-formed design (n=40, controlled, randomized, power 0.82)...")
    item strong = check_design(40, true, true, false, 0.82)
    print("  -> {} (score {})", strong.verdict, strong.score)

    print("Checking a weak design (n=12, no control, power 0.5)...")
    item weak = check_design(12, false, false, false, 0.5)
    print("  -> {} (score {})", weak.verdict, weak.score)
`,
  },
  {
    id: "massspec",
    title: "7 · Polyglot: delegate to Python",
    description:
      "kwasa-kwasa is not a replacement for other languages — it orchestrates them. " +
      "Here Turbulance hands the numeric work to a Python specialist (numpy/scipy, " +
      "running in your browser via Pyodide), then EVALUATES the result against a " +
      "hypothesis. A full task is FOUR files — you write only the .trb; the .ghd " +
      "(resources), .hre (decision log) and .fs (live state) are shown in the sidebar " +
      "so you can see the whole 'virtual brain'. First run loads Python (~7 MB).",
    code: `// Mass-spec biomarker discovery.
// Turbulance is the conductor and the judge; Python does the number-crunching.
// This task is four files: massspec.trb (logic) + .ghd (what it perceives) +
// .hre (its memory/decisions) + .fs (its live state). Only the .trb is run.

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
      "massspec.ghd": `// massspec.ghd — Gerhard dependencies (PERCEPTION)
// What this task is allowed to draw on. The framework assembles this resource
// graph; a user does not write it by hand. Shown so you can see it.

specialists:
    spectrum_analysis: "lavoisier.py"        // the Python tool the .trb delegates to

python_packages:
    - numpy
    - scipy

reference_databases:
    - hmdb:  "https://hmdb.ca"               // human metabolome database
    - kegg:  "https://rest.kegg.jp"          // metabolic pathways

ai_models:
    - scibert: "huggingface.co/allenai/scibert_scivocab_uncased"

success_criteria_from: "massspec.trb -> hypothesis BiomarkerDiscovery"
`,
      "massspec.hre": `// massspec.hre — Harare decision log (MEMORY / trajectory)
// The metacognitive trace: what the orchestrator decided and why, with
// confidence. Written by the framework as the task runs, not by the user.

session: "biomarker_discovery"
hypothesis: "Metabolomic peaks separate diabetic from control samples"

decisions:
    delegate_to_python:
        reasoning: "Peak-picking and group statistics are numeric work — hand to a specialist."
        tool: "lavoisier.py :: analyze_spectrum"
        confidence: 0.95

    evaluate_against_criteria:
        reasoning: "Turbulance judges the Python output; it does not compute it."
        criteria: ["separation >= 0.6", "num_features >= 3", "quality >= 0.7"]
        confidence: 0.90

confidence_evolution:
    - initial:           0.75
    - after_delegation:  0.90
`,
      "massspec.fs": `// massspec.fs — Fullscreen state (SENTIMENT / the task's "feel")
// A live view of where attention and resources sit. The framework updates
// this; the user reads it. (In the full system this is the sentiment field.)

flow:
    spectrum_data --> lavoisier.py (Python tool) --> structured result
                                                          |
                                                          v
                                hypothesis BiomarkerDiscovery (judge)
                                                          |
                                                          v
                                  proposition BiomarkerEvidence (verdict)

attention:
    delegate:   ###########   high   (waiting on the Python substrate)
    evaluate:   ####          low    (until results return)

status:
    python_substrate: loads-on-first-run
    verdict:          pending
`,
    },
  },
  {
    id: "paper",
    title: "8 · AI oracle: read a paper",
    description:
      "The other substrate: an AI model running in your browser (transformers.js, " +
      "no token, no backend). The oracle reads an abstract — summarize() and " +
      "classify() — and Turbulance JUDGES what it asserts via a proposition. " +
      "First run downloads small ONNX models (cached afterwards).",
    code: `// Read a paper's abstract with an in-browser model, then judge what it claims.
// The model is the oracle; Turbulance is still the judge.

hypothesis PaperReading:
    claim: "We can read an abstract and judge what it asserts"
    success_criteria:
        - assertion: 0.6

funxn main():
    item abstract = "Metformin reduced HbA1c by 1.2 percent versus placebo (p<0.001) over 24 weeks in 480 patients with type 2 diabetes. Gastrointestinal side effects were mild and transient. No serious adverse events were attributed to the drug."

    // Oracle 1: summarize the abstract
    item summary = summarize(abstract)
    print("Summary: {}", summary)

    // Oracle 2: zero-shot — what does the abstract assert?
    item efficacy = classify(abstract, ["the drug is effective", "the drug is not effective"])
    item safety = classify(abstract, ["the drug is safe", "the drug is unsafe"])
    print("Efficacy reading: {} ({})", efficacy.top, efficacy.top_score)
    print("Safety reading:   {} ({})", safety.top, safety.top_score)

    // Turbulance judges the model's reading against the hypothesis
    proposition AbstractClaims:
        motion ClaimsEfficacy("The abstract asserts the drug is effective")
        motion ClaimsSafety("The abstract asserts the drug is safe")

        given efficacy.top == "the drug is effective" and efficacy.top_score > PaperReading.success_criteria.assertion:
            support ClaimsEfficacy with_confidence(efficacy.top_score)
        given safety.top == "the drug is safe" and safety.top_score > PaperReading.success_criteria.assertion:
            support ClaimsSafety with_confidence(safety.top_score)

    print("Verdict: {}", AbstractClaims.verdict)
    return AbstractClaims
`,
  },
];

export const defaultExample = examples[0];
