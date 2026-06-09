/* transformers.js-backed AI oracle for Turbulance's oracle resolver.
 *
 * This is the browser realization of Paper B's "oracle resolver": Turbulance
 * asks a model and carries the answer's confidence (and, per the grounding
 * invariant, should check it before committing). Fully client-side —
 * @xenova/transformers downloads quantized ONNX models from the Hugging Face
 * Hub and caches them in the browser. No token, no backend. Loaded lazily. */

let transformersPromise = null;
const pipelines = {}; // "task:model" -> pipeline

async function getTransformers() {
  if (!transformersPromise) {
    transformersPromise = import("@xenova/transformers").then((mod) => {
      mod.env.allowLocalModels = false; // always fetch from the HF hub + cache
      return mod;
    });
  }
  return transformersPromise;
}

async function getPipeline(task, model, onStatus) {
  const key = `${task}:${model}`;
  if (pipelines[key]) return pipelines[key];
  const { pipeline } = await getTransformers();
  onStatus && onStatus(`Loading model ${model} …`);
  pipelines[key] = await pipeline(task, model, {
    progress_callback: (p) => {
      if (p && p.status === "progress" && p.file && typeof p.progress === "number") {
        onStatus && onStatus(`Downloading ${p.file}: ${Math.round(p.progress)}%`);
      }
    },
  });
  return pipelines[key];
}

const MODELS = {
  summarization: "Xenova/distilbart-cnn-6-6",
  zeroShot: "Xenova/nli-deberta-v3-xsmall",
  qa: "Xenova/distilbert-base-cased-distilled-squad",
};

export async function summarize(text, onStatus) {
  const pipe = await getPipeline("summarization", MODELS.summarization, onStatus);
  onStatus && onStatus("Summarizing…");
  const out = await pipe(text, { max_new_tokens: 80 });
  onStatus && onStatus(null);
  const r = Array.isArray(out) ? out[0] : out;
  return (r && (r.summary_text || r.generated_text)) || "";
}

export async function classify(text, labels, onStatus) {
  const pipe = await getPipeline("zero-shot-classification", MODELS.zeroShot, onStatus);
  onStatus && onStatus("Classifying…");
  const out = await pipe(text, labels);
  onStatus && onStatus(null);
  // out: { sequence, labels: [...sorted desc], scores: [...] }
  return {
    labels: out.labels,
    scores: out.scores.map((s) => Math.round(s * 1000) / 1000),
    top: out.labels[0],
    top_score: Math.round(out.scores[0] * 1000) / 1000,
  };
}

export async function ask(question, context, onStatus) {
  const pipe = await getPipeline("question-answering", MODELS.qa, onStatus);
  onStatus && onStatus("Reading…");
  const out = await pipe(question, context);
  onStatus && onStatus(null);
  return { answer: out.answer, score: Math.round(out.score * 1000) / 1000 };
}
