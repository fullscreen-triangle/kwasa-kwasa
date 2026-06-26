// ===========================================================================
//  TranslatorWorkbench
//
//  An AI-prompt-style interface: paste text, press Translate, and the page
//  computes the semantic-uncertainty propagation of that text and renders it as
//  sixteen D3 charts. Everything runs in the browser; there is no backend.
// ===========================================================================
import React, { useState, useCallback } from "react";
import { runTranslation } from "@/lib/translator/engine";
import TranslatorCharts from "./TranslatorCharts";

const SAMPLE =
  "A medical student at Harvard reports that the new protocol improves recovery. " +
  "However, the sample was small and the follow-up period was short. " +
  "The committee approved the study, yet the loan that funded it remains under review.";

function Stat({ label, value }) {
  return (
    <div className="flex flex-col rounded-md border border-dark/10 bg-white px-3 py-2 dark:border-light/15">
      <span className="text-[10px] uppercase tracking-wide text-dark/50 dark:text-light/50">{label}</span>
      <span className="text-sm font-semibold text-dark dark:text-light">{value}</span>
    </div>
  );
}

export default function TranslatorWorkbench() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  const translate = useCallback(async () => {
    const input = text.trim();
    if (!input) {
      setError("Paste some text first.");
      return;
    }
    setError(null);
    setRunning(true);
    setResult(null);
    // yield a frame so the loading state paints before the (synchronous) work
    await new Promise((r) => setTimeout(r, 30));
    try {
      const r = runTranslation(input);
      setResult(r);
    } catch (e) {
      setError(e && e.message ? e.message : "Something went wrong.");
    } finally {
      setRunning(false);
    }
  }, [text]);

  const onKeyDown = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      translate();
    }
  };

  return (
    <div className="flex w-full flex-col gap-6">
      {/* ---- prompt box ------------------------------------------------ */}
      <div className="mx-auto w-full max-w-3xl">
        <div className="rounded-2xl border border-dark/15 bg-white p-3 shadow-sm dark:border-light/20 dark:bg-dark/40">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Paste the text you want to translate…"
            className="h-44 w-full resize-y bg-transparent px-2 py-1 text-sm text-dark outline-none placeholder:text-dark/40 dark:text-light dark:placeholder:text-light/40"
          />
          <div className="mt-2 flex items-center justify-between border-t border-dark/10 pt-2 dark:border-light/15">
            <div className="flex items-center gap-3 text-[11px] text-dark/50 dark:text-light/50">
              <button
                type="button"
                onClick={() => setText(SAMPLE)}
                className="rounded border border-dark/15 px-2 py-1 font-medium hover:bg-dark/5 dark:border-light/20 dark:hover:bg-light/10"
              >
                Use sample
              </button>
              <span>{text.trim() ? text.trim().split(/\s+/).length : 0} words</span>
              <span className="hidden sm:inline">⌘/Ctrl + Enter to run</span>
            </div>
            <button
              type="button"
              onClick={translate}
              disabled={running}
              className="rounded-lg bg-dark px-5 py-2 text-sm font-semibold text-light transition hover:opacity-90 disabled:opacity-50 dark:bg-light dark:text-dark"
            >
              {running ? "Translating…" : "Translate"}
            </button>
          </div>
        </div>
        {error && <p className="mt-2 text-center text-xs font-medium text-red-600">{error}</p>}
      </div>

      {/* ---- loading --------------------------------------------------- */}
      {running && (
        <div className="flex flex-col items-center justify-center gap-3 py-16 text-dark/60 dark:text-light/60">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-dark/20 border-t-dark dark:border-light/20 dark:border-t-light" />
          <p className="text-sm">Propagating uncertainty over the contact graph…</p>
        </div>
      )}

      {/* ---- results --------------------------------------------------- */}
      {result && !running && (
        <div className="flex flex-col gap-6">
          <div className="grid grid-cols-4 gap-3 md:grid-cols-2">
            <Stat label="Tokens" value={result.summary.tokens} />
            <Stat label="Vocabulary (items)" value={result.summary.vocab} />
            <Stat label="Floor β" value={result.summary.floor.toFixed(2)} />
            <Stat label="Min σ (text)" value={result.summary.minSigma.toFixed(2)} />
            <Stat label="Total weight Ω" value={result.summary.omega.toFixed(1)} />
            <Stat label="Relational density" value={result.summary.relationalDensity.toFixed(2)} />
            <Stat label="False-friend rate" value={(result.summary.falseFriendRate * 100).toFixed(0) + "%"} />
            <Stat label="Relaxation" value={result.summary.quiescent ? "quiescent" : "non-halting"} />
          </div>
          <TranslatorCharts result={result} />
        </div>
      )}

      {/* ---- empty state ----------------------------------------------- */}
      {!result && !running && (
        <p className="mx-auto max-w-2xl text-center text-sm text-dark/50 dark:text-light/50">
          The translation is reported as sixteen charts derived directly from
          your text: the resolution floor on its tokens, the individuation and
          identity structure, the propagation and relaxation to quiescence, and
          the four-column route audit. All computed in your browser by exact
          minimum cut.
        </p>
      )}
    </div>
  );
}
