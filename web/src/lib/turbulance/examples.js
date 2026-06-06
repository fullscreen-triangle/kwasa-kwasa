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
];

export const defaultExample = examples[0];
