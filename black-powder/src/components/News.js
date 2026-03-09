import { Fragment, useEffect, useState } from "react";
import Modal from "react-modal";
const News = ({ ActiveIndex, animation }) => {
  const [isOpen4, setIsOpen4] = useState(false);
  const [modalContent, setModalContent] = useState({});

  useEffect(() => {
    var lists = document.querySelectorAll(".news_list > ul > li");
    let box = document.querySelector(".cavani_fn_moving_box");
    if (!box) {
      let body = document.querySelector("body");
      let div = document.createElement("div");
      div.classList.add("cavani_fn_moving_box");
      body.appendChild(div);
    }

    lists.forEach((list) => {
      list.addEventListener("mouseenter", (event) => {
        box.classList.add("opened");
        var imgURL = list.getAttribute("data-img");
        box.style.backgroundImage = `url(${imgURL})`;
        box.style.top = event.clientY - 50 + "px";
        if (imgURL === "") {
          box.classList.remove("opened");
          return false;
        }
      });
      list.addEventListener("mouseleave", () => {
        box.classList.remove("opened");
      });
    });
  }, []);

  function toggleModalFour(value) {
    setIsOpen4(!isOpen4);
    setModalContent(value);
  }
  const newsData = [
    {
      img: "img/figures/panel_4_crossscale.png",
      tag: "Research Paper",
      date: "2025",
      comments: "Preprint",
      title: "Categorical Apertures as Semantic Computation Primitives: A Unified Topological Framework for Biological Information Filtering",
      text1:
        "We introduce categorical apertures — zero-work topological filters that provide a unified mathematical framework for understanding biological information processing across 13 orders of magnitude. The framework is built on three axioms (closure, composition, identity) and demonstrates that a single entropy equation S = k_B M ln(n) governs information filtering at molecular, neural, and pharmacological scales.",
      text2:
        "Three independent validation layers confirm the framework's predictions: (1) Molecular scale — enzyme kinetics from KEGG with d_cat-efficiency correlation r=0.902; (2) Neural scale — EEG phase-locking with PLV-Kuramoto correlation r=0.988; (3) Pharmacological scale — ChEMBL drug binding profiles mapping to predicted S-entropy coordinates.",
      text3:
        "The framework identifies five operational regimes (Coherent, Phase-Locked, Hierarchical Cascade, Aperture-Dominated, Turbulent) and demonstrates clinical relevance through depression treatment analysis: PLV recovery from 0.32 to 0.77, HAM-D reduction from 24.3 to 8.5, and 83% treatment response rate.",
    },
    {
      img: "img/figures/panel_1_molecular.png",
      tag: "Validation Code",
      date: "2025",
      comments: "Open Source",
      title: "Semantic Maxwell Demon: Three-Layer Validation Suite",
      text1:
        "Complete Python validation suite implementing all three validation layers of the categorical aperture framework. Includes molecular validation (KEGG enzyme kinetics, ETC complex analysis), neural validation (EEG phase-locking, Kuramoto synchronization), and pharmacological validation (ChEMBL drug binding profiles).",
      text2:
        "The validation code generates all 5 panel figures from the paper and includes statistical analysis pipelines for correlation testing, regime classification, and cross-scale unification metrics.",
      text3:
        "Built with NumPy, SciPy, and Matplotlib. Available on GitHub with full documentation and reproducible analysis notebooks.",
    },
    {
      img: "img/figures/panel_2_neural.png",
      tag: "Dataset",
      date: "2025",
      comments: "Multi-Source",
      title: "Cross-Scale Validation Datasets: KEGG, ChEMBL, OpenNeuro",
      text1:
        "The validation leverages publicly available datasets spanning three biological scales. KEGG provides enzyme kinetics parameters (k_cat, K_m) for 6 major enzyme classes. ChEMBL provides drug binding affinities (Ki, IC50) for 8 antidepressant medications across 5 receptor targets.",
      text2:
        "Neural data uses EEG recordings from OpenNeuro and PhysioNet, analyzing phase-locking values across theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), and gamma (30-100 Hz) frequency bands in healthy controls and depressed subjects.",
      text3:
        "All datasets are freely accessible and the analysis pipeline is fully reproducible. The cross-scale unification demonstrates that identical mathematical structures emerge independently at each scale.",
    },
  ];
  return (
    <Fragment>
      <div
        className={
          ActiveIndex === 3
            ? `cavani_tm_section active animated ${animation ? animation : "fadeInUp"
            }`
            : "cavani_tm_section hidden animated"
        }
        id="news__"
      >
        <div className="section_inner">
          <div className="cavani_tm_news">
            <div className="cavani_tm_title">
              <span>Publications & Resources</span>
            </div>
            <div className="news_list">
              <ul>
                {newsData.map((news, i) => (
                  <li data-img={news.img} key={i}>
                    <div className="list_inner">
                      <span className="number">{`${i <= 9 ? 0 : ""}${i + 1
                        }`}</span>
                      <div className="details">
                        <div className="extra_metas">
                          <ul>
                            <li>
                              <span>{news.date}</span>
                            </li>
                            <li>
                              <span>
                                <a
                                  href="#"
                                  onClick={() => toggleModalFour(news)}
                                >
                                  {news.tag}
                                </a>
                              </span>
                            </li>
                            <li>
                              <span>
                                <a
                                  href="#"
                                  onClick={() => toggleModalFour(news)}
                                >
                                  {news.comments}
                                </a>
                              </span>
                            </li>
                          </ul>
                        </div>
                        <div className="post_title">
                          <h3>
                            <a href="#" onClick={() => toggleModalFour(news)}>
                              {news.title}
                            </a>
                          </h3>
                        </div>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
      {modalContent && (
        <Modal
          isOpen={isOpen4}
          onRequestClose={toggleModalFour}
          contentLabel="My dialog"
          className="mymodal"
          overlayClassName="myoverlay"
          closeTimeoutMS={300}
          openTimeoutMS={300}
        >
          <div className="cavani_tm_modalbox opened">
            <div className="box_inner">
              <div className="close" onClick={toggleModalFour}>
                <a href="#">
                  <i className="icon-cancel"></i>
                </a>
              </div>
              <div className="description_wrap">
                <div className="news_popup_informations">
                  <div className="image">
                    <img src="img/thumbs/4-2.jpg" alt="" />
                    <div
                      className="main"
                      data-img-url={modalContent.img}
                      style={{ backgroundImage: `url(${modalContent.img})` }}
                    />
                  </div>
                  <div className="details">
                    <div className="meta">
                      <ul>
                        <li><span>{modalContent.date}</span></li>
                        <li><span><a href="#">{modalContent.tag}</a></span></li>
                        <li><span><a href="#">{modalContent.comments}</a></span></li>
                      </ul>
                    </div>
                    <div className="title">
                      <h3>{modalContent.title}</h3>
                    </div>
                  </div>
                  <div className="text">
                    <p>{modalContent.text1}</p>
                    <p>{modalContent.text2}</p>
                    <p>{modalContent.text3}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Modal>
      )}
    </Fragment>
  );
};
export default News;
