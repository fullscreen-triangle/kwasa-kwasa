import Head from "next/head";
import dynamic from "next/dynamic";
import TransitionEffect from "@/components/TransitionEffect";

// The workbench renders SVG via D3 against measured DOM, so it is client-only.
const TranslatorWorkbench = dynamic(
  () => import("@/components/TranslatorWorkbench"),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-[60vh] w-full items-center justify-center text-dark dark:text-light">
        Loading translator…
      </div>
    ),
  }
);

export default function TranslatorPage() {
  return (
    <>
      <Head>
        <title>Translator | kwasa-kwasa</title>
        <meta
          name="description"
          content="Paste text and see its semantic-uncertainty propagation reported as sixteen D3 charts — the resolution floor, individuation, propagation, and the four-column route audit, computed in your browser."
        />
      </Head>
      <TransitionEffect />
      <article className="flex min-h-screen w-full flex-col items-center text-dark dark:text-light">
        <div className="w-full px-10 pt-6 pb-16 lg:px-8 md:px-6 sm:px-4">
          <h1 className="mb-1 text-3xl font-bold sm:text-2xl">Translator</h1>
          <p className="mb-6 max-w-3xl text-sm font-medium text-dark/70 dark:text-light/70">
            Paste a passage and translate it into its semantic-uncertainty
            propagation. The text is terminated into a contact graph (tokens as
            items, the rest as the medium), and the same constructions behind the
            paper&apos;s sixteen figures are run over it — the resolution floor,
            identity as a region, the relaxation to quiescence, and the
            four-column route audit. Everything runs locally; nothing is sent to
            a server.
          </p>
          <TranslatorWorkbench />
        </div>
      </article>
    </>
  );
}
