import Head from "next/head";
import dynamic from "next/dynamic";

// Three.js / r3f must render client-side only.
const GardenScene = dynamic(() => import("@/components/GardenScene"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center text-dark dark:text-light">
      loading…
    </div>
  ),
});

export default function Home() {
  return (
    <>
      <Head>
        <title>kwasa-kwasa</title>
        <meta
          name="description"
          content="kwasa-kwasa — a semantic computing framework and its language, Turbulance."
        />
      </Head>

      <section className="relative h-[calc(100vh-7rem)] w-full overflow-hidden">
        <div className="pointer-events-none absolute inset-x-0 top-4 z-10 flex justify-center px-4">
          <h1 className="text-center font-bold tracking-tight text-dark dark:text-light text-7xl xl:text-6xl md:text-5xl sm:text-4xl">
            kwasa-kwasa
          </h1>
        </div>
        <GardenScene />
      </section>
    </>
  );
}
