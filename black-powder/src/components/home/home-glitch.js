import Link from "next/link";
import React from "react";
import { RotateTextAnimation } from "../AnimationText";

export default function HomeGlitch({ ActiveIndex, handleOnClick }) {
  return (
    <>
      {/* <!-- HOME --> */}
      <div
        className={
          ActiveIndex === 0
            ? "cavani_tm_section animated rollIn"
            : "cavani_tm_section animated hidden rollOut"
        }
        id="home_"
      >
        <div className="cavani_tm_home">
          <div className="content">
            <h3 className="name">Kelly Emerson</h3>
            <span className="line" />
            <h3 className="job">
              <RotateTextAnimation />
            </h3>
            <div className="cavani_tm_button transition_link">
              <Link href="#about">
                <a onClick={() => handleOnClick(4)}>Learn More</a>
              </Link>
            </div>
          </div>
        </div>
      </div>
      {/* HOME */}
    </>
  );
}
