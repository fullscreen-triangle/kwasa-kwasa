import Link from "next/link";
import React from "react";
import { LoadingTextAnimation } from "../AnimationText";

export default function HomeDefault({ ActiveIndex, handleOnClick }) {
  return (
    <>
      {/* <!-- HOME --> */}
      <div
        className={
          ActiveIndex === 0
            ? "cavani_tm_section active animated flipInX"
            : "cavani_tm_section active hidden animated flipOutX"
        }
        id="home_"
      >
        <div className="cavani_tm_home">
          <div className="content">
            <h3 className="name">Rosa Vincent</h3>
            <span className="line"></span>
            <h3 className="job">
              <LoadingTextAnimation />
            </h3>
            <div className="cavani_tm_button transition_link">
              <Link href="#contact">
                <a onClick={() => handleOnClick(4)}>Get in Touch</a>
              </Link>
            </div>
          </div>
        </div>
      </div>
      {/* <!-- HOME --> */}
    </>
  );
}
