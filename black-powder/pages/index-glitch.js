import dynamic from "next/dynamic";
import React, { useState } from "react";
import AuthorDefault from "../src/components/author/author-glitch";
import ContactGlitch from "../src/components/contact/contact-glitch";
import HomeGlitch from "../src/components/home/home-glitch";
import News from "../src/components/News";
import Service from "../src/components/service/service-glitch";
import Header from "../src/layout/header";
import Layout from "../src/layout/layout";
import LeftRightBar from "../src/layout/left-right-bar";
import Mobilemenu from "../src/layout/mobilemenu";
import Modalbox from "../src/layout/modalbox";
import TopBar from "../src/layout/top-bar";


const AboutGlitch = dynamic(
  () => import("../src/components/about/about-glitch"),
  {
    ssr: false,
  }
);
const PortfolioGlitch = dynamic(
  () => import("../src/components/portfolio/portfolio-default"),
  {
    ssr: false,
  }
);

export default function Home() {
  const [ActiveIndex, setActiveIndex] = useState(0);
  const handleOnClick = (index) => {
    setActiveIndex(index); // remove the curly braces
  };

  const [isToggled, setToggled] = useState(false);
  const toggleTrueFalse = () => setToggled(!isToggled);

  return (
    <>
      <Layout>
        <div
          className="cavani_tm_all_wrap"
          data-magic-cursor="show"
          data-enter="rollIn"
          data-exit="rollOut"
        >
          <Modalbox />
          <Header handleOnClick={handleOnClick} ActiveIndex={ActiveIndex} />
          <LeftRightBar />
          <TopBar toggleTrueFalse={toggleTrueFalse} isToggled={isToggled} />
          <Mobilemenu toggleTrueFalse={toggleTrueFalse} isToggled={isToggled} handleOnClick={handleOnClick} />

          {/* MAINPART */}
          <div className="cavani_tm_mainpart">
            <AuthorDefault />
            <div className="main_content">
              <HomeGlitch ActiveIndex={ActiveIndex} handleOnClick={handleOnClick} />
              <AboutGlitch ActiveIndex={ActiveIndex} />
              <PortfolioGlitch Animation={"rollIn"} Animation2={"rollOut"} ActiveIndex={ActiveIndex} />
              <Service ActiveIndex={ActiveIndex} />
              <News animation={"rollIn "} ActiveIndex={ActiveIndex} />
              <ContactGlitch ActiveIndex={ActiveIndex} />
            </div>
          </div>
          {/* /MAINPART */}

        </div>
        {/* / WRAPPER ALL */}
      </Layout>
    </>
  );
}
