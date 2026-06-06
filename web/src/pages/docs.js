import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import Head from "next/head";
import Link from "next/link";

const Section = ({ title, children }) => (
  <section className="mb-10">
    <h2 className="mb-3 text-2xl font-bold sm:text-xl">{title}</h2>
    <div className="text-base font-medium text-dark/80 dark:text-light/80 md:text-sm">
      {children}
    </div>
  </section>
);

const Code = ({ children }) => (
  <pre className="my-4 overflow-x-auto rounded-lg border-2 border-dark/20 bg-dark p-4 font-mono text-[13px] leading-relaxed text-light dark:border-light/20 sm:text-[11px]">
    {children}
  </pre>
);

export default function Docs() {
  return (
    <>
      <Head>
        <title>Docs | kwasa-kwasa</title>
        <meta
          name="description"
          content="An introduction to Turbulance — propositions, motions, points, resolution, and the confidence algebra."
        />
      </Head>
      <TransitionEffect />
      <article className="flex min-h-screen flex-col items-center text-dark dark:text-light">
        <Layout className="!pt-16">
          <AnimatedText
            text="Turbulance, in brief."
            className="mb-12 !text-6xl lg:!text-5xl sm:!text-3xl"
          />

          <div className="mx-auto max-w-3xl">
            <Section title="What it is">
              Turbulance is the language of the kwasa-kwasa framework. Unlike a
              conventional language, its primitive notions are not values and
              side effects but <em>evidence</em>, <em>hypotheses</em>, and{" "}
              <em>graded uncertainty</em>. A program does not script a sequence
              of steps; it declares what it is trying to establish and
              accumulates support for it. The full formal specification — its
              grammar, type system, and operational semantics — is given in the
              accompanying paper; this page is a five-minute tour.
            </Section>

            <Section title="Functions and bindings">
              <code>funxn</code> declares a function (not <code>function</code>);{" "}
              <code>item</code> binds a value; <code>given</code> is the
              conditional (not <code>if</code>). Blocks are indented.
              <Code>{`funxn greet(name):
    item message = "Hello, " + name
    print(message)
    return message`}</Code>
            </Section>

            <Section title="Propositions and motions">
              A <code>proposition</code> is a first-class hypothesis, decomposed
              into named <code>motion</code>s. <code>support</code> and{" "}
              <code>contradict</code> accumulate graded evidence; each motion
              receives a confidence in [0, 1] and a verdict — Supported,
              Contradicted, or Inconclusive.
              <Code>{`proposition DrugEfficacy:
    motion ReducesSymptoms("The drug reduces symptom severity")

    given true:
        support ReducesSymptoms with_confidence(0.8)
        support ReducesSymptoms with_confidence(0.7)`}</Code>
              Corroborating evidence combines by the noisy-or rule{" "}
              <code>c ⊕ d = 1 − (1 − c)(1 − d)</code>, so confidence accumulates
              but never exceeds certainty.
            </Section>

            <Section title="Points and resolution">
              A <code>point</code> is a datum that carries its own confidence
              and (optionally) a distribution over interpretations.{" "}
              <code>resolve</code> collapses a point to a determinate{" "}
              <em>entity</em> under an explicit strategy.
              <Code>{`point diagnosis = {
    content: "elevated marker, ambiguous aetiology",
    confidence: 0.7
}

item determined = resolve diagnosis`}</Code>
            </Section>

            <Section title="Try it">
              Every construct above runs in the{" "}
              <Link href="/playground" className="font-semibold underline">
                Playground
              </Link>{" "}
              — pick a tutorial from the sidebar, edit it, and press Run. The
              browser version implements the deterministic core of the language;
              the full runtime, with external resolvers and the sentiment field,
              is specified in the companion papers and implemented in Rust.
            </Section>
          </div>
        </Layout>
      </article>
    </>
  );
}
