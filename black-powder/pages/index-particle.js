import dynamic from "next/dynamic";
import React, { useState } from "react";
import ContactDefault from "../src/components/contact/contact-particle";
import HomeDefault from "../src/components/home/home-particle";
import Layout from "../src/layout/layout";
import News from "../src/components/News";
import Header from "../src/layout/header";
import LeftRightBar from "../src/layout/left-right-bar";
import Mobilemenu from "../src/layout/mobilemenu";
import Modalbox from "../src/layout/modalbox";
import TopBar from "../src/layout/top-bar";
import Service from "../src/components/service/service-particle";
// import AboutDefault from "../src/components/about/about-particle";
// import AuthorDefault from "../src/components/author/author-particle";

const AboutDefault = dynamic(
  () => import("../src/components/about/about-particle"),
  {
    ssr: false,
  }
);
const AuthorDefault = dynamic(
  () => import("../src/components/author/author-particle"),
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
              <HomeDefault  ActiveIndex={ActiveIndex}  handleOnClick={handleOnClick} />

              <AboutDefault ActiveIndex={ActiveIndex} />

              <PortfolioDefault Animation={"flipInX"} ActiveIndex={ActiveIndex} />

              <Service ActiveIndex={ActiveIndex} />

              <News animation={"flipInX"} ActiveIndex={ActiveIndex} />

              <ContactDefault ActiveIndex={ActiveIndex} />
            </div>
          </div>
          {/* MAINPART */}
          
        </Layout>
      </>
    );
}
