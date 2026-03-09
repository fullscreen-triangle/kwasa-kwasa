import { Fragment, useEffect, useState } from "react";
import TypeAnimation from "react-type-animation";
export const AnimationText1 = () => {
  return (
    <Fragment>
      <TypeAnimation
        cursor={true}
        sequence={[" Zero-Work Filtering", 1000, " Triple Equivalence", 1000, " 13 Orders of Magnitude", 1000]}
        wrapper="span"
        repeat={Infinity}
      />
    </Fragment>
  );
};

const skills = ["Zero-Work Filtering", "Triple Equivalence", "13 Orders of Magnitude", "Five Operational Regimes"];
export const RotateTextAnimation = () => {
  const [text, setText] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => {
      setText(text < skills.length - 1 ? text + 1 : 0);
    }, 5000);
    return () => clearInterval(interval);
  });
  return (
    <Fragment>
      <span className="cd-headline rotate-1">
        <span className="blc">Semantic </span>
        <span className="cd-words-wrapper">
          {skills.map((skill, i) => (
            <b key={i} className={text === i ? "is-visible" : "is-hidden"}>
              {skill}
            </b>
          ))}
        </span>
      </span>
    </Fragment>
  );
};
export const ZoomTextAnimation = () => {
  const [text, setText] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => {
      setText(text < skills.length - 1 ? text + 1 : 0);
    }, 5000);
    return () => clearInterval(interval);
  });
  return (
    <Fragment>
      <span className="cd-headline zoom">
        <span className="blc">Semantic </span>
        <span className="cd-words-wrapper">
          {skills.map((skill, i) => (
            <b key={i} className={text === i ? "is-visible" : "is-hidden"}>
              {skill}
            </b>
          ))}
        </span>
      </span>
    </Fragment>
  );
};
export const LoadingTextAnimation = () => {
  const [text, setText] = useState(0);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const interval = setInterval(() => {
      setText(text < skills.length - 1 ? text + 1 : 0);
      setLoading(!loading);
    }, 5000);
    if (loading == false) {
      setTimeout(() => {
        setLoading(true);
      }, 100);
    }
    return () => clearInterval(interval);
  });
  return (
    <Fragment>
      <span className="cd-headline loading-bar">
        <span className="blc">Semantic </span>
        <span className={`cd-words-wrapper ${loading ? "is-loading" : ""}`}>
          {skills.map((skill, i) => (
            <b key={i} className={text === i ? "is-visible" : "is-hidden"}>
              {skill}
            </b>
          ))}
        </span>
      </span>
    </Fragment>
  );
};
