import Head from "next/head";
import React from "react";
import Preloader from "../components/Preloader";
import Footer from "./footer";
import MagicCursor from "./magic-cursor";

export default function Layout({ children }) {
  return (
    <>
      <Head>

        <meta name="description" content="Categorical Apertures: A unified topological framework for biological information filtering across 13 orders of magnitude, from molecular enzymes to neural consciousness." />
        <meta name="author" content="Kundai F. Sachikonye" />
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />

        <title>Categorical Apertures | Semantic Framework</title>

        {/* <!-- STYLES --> */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="true" />
        <link href="https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,300;1,400;1,500;1,600;1,700;1,800&display=swap" rel="stylesheet" />
        <link href="https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" rel="stylesheet" />
        
      </Head>
      <Preloader />
      <div className="cavani_tm_all_wrap">
        {children}
        <Footer />
        <MagicCursor />
      </div>
      <div className="cavani_fn_moving_box"></div>
    </>
  );
}
