import { Fragment, useEffect } from "react";
const Preloader = () => {
  useEffect(() => {
    let isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry/i.test(
      navigator.userAgent
    )
      ? true
      : false;
    let preloader = document.getElementById("preloader");

    if (preloader) {
      if (!isMobile) {
        setTimeout(function () {
          preloader.classList.add("preloaded");
        }, 800);
        setTimeout(function () {
          preloader.remove();
        }, 2000);
      } else {
        preloader.remove();
      }
    }
    setTimeout(() => {
      document.querySelector("body").classList.add("opened");
    }, 3000);
  }, []);

  return (
    <Fragment>
      <div id="preloader">
        <div className="loader_line" />
      </div>
    </Fragment>
  );
};
export default Preloader;
