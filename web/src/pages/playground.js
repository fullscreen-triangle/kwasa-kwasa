import Head from "next/head";
import dynamic from "next/dynamic";
import TransitionEffect from "@/components/TransitionEffect";

// The playground touches window/document, so render it client-side only.
const TurbulancePlayground = dynamic(
  () => import("@/components/TurbulancePlayground"),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-[78vh] w-full items-center justify-center text-dark dark:text-light">
        Loading playground…
      </div>
    ),
  }
);

export default function PlaygroundPage() {
  return (
    <>
      <Head>
        <title>Playground | kwasa-kwasa</title>
        <meta
          name="description"
          content="Run Turbulance scripts in your browser — propositions, motions, points, and graded confidence, with no backend."
        />
      </Head>
      <TransitionEffect />
      <article className="flex min-h-screen w-full flex-col items-center text-dark dark:text-light">
        <div className="w-full px-10 pt-6 pb-10 lg:px-8 md:px-6 sm:px-4">
          <h1 className="mb-1 text-3xl font-bold sm:text-2xl">Turbulance Playground</h1>
          <p className="mb-5 max-w-3xl text-sm font-medium text-dark/70 dark:text-light/70">
            Write small Turbulance scripts and run them entirely in your
            browser — propositions and motions with graded support, points and
            resolution, the confidence algebra. Pick a tutorial from the
            sidebar, edit, and press{" "}
            <span className="font-semibold">Run</span> (or Ctrl/⌘+Enter).
          </p>
          <div className="h-[78vh] min-h-[520px] w-full">
            <TurbulancePlayground />
          </div>
        </div>
      </article>
    </>
  );
}
