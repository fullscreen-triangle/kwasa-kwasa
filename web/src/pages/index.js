import AnimatedText from "@/components/AnimatedText";
import { LinkArrow } from "@/components/Icons";
import Layout from "@/components/Layout";
import Head from "next/head";
import Link from "next/link";
import TransitionEffect from "@/components/TransitionEffect";

export default function Home() {
  return (
    <>
      <Head>
        <title>kwasa-kwasa — the Turbulance language</title>
        <meta
          name="description"
          content="kwasa-kwasa is a semantic computing framework. Its language, Turbulance, makes propositions, motions, points, and graded confidence first-class. Run it in your browser."
        />
      </Head>

      <TransitionEffect />
      <article className="flex min-h-screen items-center justify-center text-dark dark:text-light">
        <Layout className="!pt-0 md:!pt-16 sm:!pt-16">
          <div className="flex w-full flex-col items-center text-center">
            <AnimatedText
              text="A language that reasons under evidence."
              className="!text-6xl xl:!text-5xl lg:!text-6xl md:!text-5xl sm:!text-3xl"
            />
            <p className="my-6 max-w-2xl text-base font-medium md:text-sm sm:!text-xs">
              <span className="font-semibold">kwasa-kwasa</span> is a semantic
              computing framework. Its language,{" "}
              <span className="font-semibold">Turbulance</span>, makes
              propositions, motions, points, and graded confidence first-class —
              so programs accumulate evidence toward a conclusion instead of
              scripting steps. Try it in your browser, no install required.
            </p>
            <div className="mt-2 flex items-center gap-4 sm:flex-col">
              <Link
                href="/playground"
                className="flex items-center rounded-lg border-2 border-solid bg-dark p-2.5 px-6 text-lg font-semibold
                capitalize text-light hover:border-dark hover:bg-transparent hover:text-dark
                dark:bg-light dark:text-dark dark:hover:border-light dark:hover:bg-dark dark:hover:text-light
                md:p-2 md:px-4 md:text-base"
              >
                Open the Playground <LinkArrow className="ml-1 !w-6 md:!w-4" />
              </Link>
              <Link
                href="/docs"
                className="text-lg font-medium capitalize text-dark underline dark:text-light md:text-base"
              >
                Read the docs
              </Link>
            </div>

            <pre className="mt-12 max-w-xl overflow-x-auto rounded-lg border-2 border-dark/30 bg-dark p-5 text-left font-mono text-[13px] leading-relaxed text-light dark:border-light/30 sm:text-[11px]">
{`proposition DataQuality:
    motion CompleteData("Dataset has no missing values")

    within "dataset":
        given missing_count == 0:
            support CompleteData with_confidence(0.95)`}
            </pre>
          </div>
        </Layout>
      </article>
    </>
  );
}
