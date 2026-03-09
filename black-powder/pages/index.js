import dynamic from "next/dynamic";
import React, { useState } from "react";
import HomeDefault from "../src/components/home/home-default";
import Layout from "../src/layout/layout";
// import AuthorDefault from "../src/components/author/author-default";
import ContactDefault from "../src/components/contact/contact-default";
import News from "../src/components/News";
import Header from "../src/layout/header";
import LeftRightBar from "../src/layout/left-right-bar";
import Mobilemenu from "../src/layout/mobilemenu";
import Modalbox from "../src/layout/modalbox";
import TopBar from "../src/layout/top-bar";
import Service from "../src/components/service/service-default";

const AuthorDefault = dynamic(
  () => import("../src/components/author/author-water"),
  { ssr: false }
);
const AboutDefault = dynamic(
  () => import("../src/components/about/about-default"),
  {
    ssr: false,
  }
);
const PortfolioDefault = dynamic(
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
        <Modalbox />
        <Header handleOnClick={handleOnClick} ActiveIndex={ActiveIndex} />
        <LeftRightBar />
        <TopBar toggleTrueFalse={toggleTrueFalse} isToggled={isToggled} />
        <Mobilemenu toggleTrueFalse={toggleTrueFalse} isToggled={isToggled} handleOnClick={handleOnClick} />

        {/* <!-- MAINPART --> */}
        <div className="cavani_tm_mainpart">
          <AuthorDefault />

          <div className="main_content">
            <HomeDefault ActiveIndex={ActiveIndex} handleOnClick={handleOnClick} />

            <AboutDefault ActiveIndex={ActiveIndex} />

            <PortfolioDefault ActiveIndex={ActiveIndex} />

            <Service ActiveIndex={ActiveIndex} />

            <News ActiveIndex={ActiveIndex} />

            <ContactDefault ActiveIndex={ActiveIndex} />
          </div>
        </div>
        {/* MAINPART */}
      </Layout>
    </>
  );
}
