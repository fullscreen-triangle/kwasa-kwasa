import "particles.js/particles";
import React, { useEffect } from "react";
export default function AuthorDefault() {
  useEffect(() => {
    const particlesJS = window.particlesJS;
    particlesJS.load("particles-js", "particlesConfig.json", function () {
      console.log("hi");
    });
  }, []);

  return (
    <>
      <div className="author_image">
        <div
          className="main"
          data-img-url="img/about/4.jpg"
          style={{ backgroundImage: `url(img/about/4.jpg)` }}
        ></div>
        <div className="particle_wrapper">
          <div id="particles-js" />
        </div>
      </div>
    </>
  );
}
