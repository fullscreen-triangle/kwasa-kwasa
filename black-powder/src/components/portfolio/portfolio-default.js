import { useState, useEffect, useRef } from 'react'
import Isotope from 'isotope-layout'
import Modal from 'react-modal';

export default function PortfolioDefault({ ActiveIndex, Animation }) {

    const [isOpen4, setIsOpen4] = useState(false);
    const [modalContent, setModalContent] = useState({});

    function toggleModal(item) {
        setModalContent(item);
        setIsOpen4(!isOpen4);
    }

    function closeModal() {
        setIsOpen4(false);
    }

    const figures = [
        {
            category: "molecular",
            img: "img/figures/panel_1_molecular.png",
            title: "Panel 1: Molecular Scale",
            caption: "(A) 3D S-entropy coordinate mapping of enzyme classes. (B) Categorical distance d_cat vs catalytic efficiency showing r=0.902 correlation. (C) Electron transport chain flux through Complexes I-V. (D) Timescale hierarchy from femtosecond bond vibrations to millisecond catalytic turnover."
        },
        {
            category: "neural",
            img: "img/figures/panel_2_neural.png",
            title: "Panel 2: Neural Scale",
            caption: "(A) 3D (R, sigma^2, S) phase space with regime boundaries. (B) Kuramoto order parameter R across EEG frequency bands. (C) Phase-locking value vs Kuramoto R showing r=0.988 correlation. (D) Regime occupancy heatmap for healthy vs depressed subjects."
        },
        {
            category: "pharmacological",
            img: "img/figures/panel_3_pharma.png",
            title: "Panel 3: Pharmacological Scale",
            caption: "(A) 3D S-entropy mapping of antidepressant drug profiles. (B) Multi-receptor binding profiles (5-HT, NET, DAT, sigma-1, NMDA). (C) Clinical response rates with aperture boundary markers. (D) Selectivity index vs categorical distance d_cat."
        },
        {
            category: "crossscale",
            img: "img/figures/panel_4_crossscale.png",
            title: "Panel 4: Cross-Scale Unification",
            caption: "(A) 3D regime boundary surface across all scales. (B) Regime boundaries contour plot in (R, sigma^2) plane. (C) log10(S) entropy values across molecular, neural, and pharmacological scales. (D) Triple equivalence S_osc = S_cat = S_part scatter with unity line."
        },
        {
            category: "crossscale",
            img: "img/figures/panel_5_dynamics.png",
            title: "Panel 5: Dynamic Trajectories",
            caption: "(A) 3D variance landscape showing energy minimization paths. (B) Order parameter R(t) trajectory during treatment response. (C) Poincare section of oscillatory dynamics at gamma frequency. (D) Regime occupancy evolution as stacked time series."
        }
    ];

    // init one ref to store the future isotope object
    const isotope = useRef()
    // store the filter keyword in a state
    const [filterKey, setFilterKey] = useState('*')

    // initialize an Isotope object with configs
    useEffect(() => {
        setTimeout(() => {
            isotope.current = new Isotope(".filter-container", {
                itemSelector: ".filter-item",
                   layoutMode: "fitRows",
            });
        }, 500);
        return () => isotope.current && isotope.current.destroy();
    }, []);

    // handling filter key change
    useEffect(() => {
        if (isotope.current) {
            filterKey === '*'
                ? isotope.current.arrange({ filter: '*' })
                : isotope.current.arrange({ filter: `.${filterKey}` })
        }
    }, [filterKey])

    const handleFilterKeyChange = key => () => setFilterKey(key)

    return (
        <>
            {/* <!-- FIGURES --> */}

            <div className={ActiveIndex === 2 ? `cavani_tm_section active animated ${Animation ? Animation: "fadeInUp"}` : "cavani_tm_section hidden animated"} id="portfolio_">
                <div className="section_inner">
                    <div className="cavani_tm_portfolio">
                        <div className="cavani_tm_title">
                            <span>Figures & Visualizations</span>
                        </div>

                        <div className="portfolio_filter">
                            <ul>
                                <li><a href='#' onClick={handleFilterKeyChange('*')} className="current">All</a></li>
                                <li><a href='#' onClick={handleFilterKeyChange('molecular')}>Molecular</a></li>
                                <li><a href='#' onClick={handleFilterKeyChange('neural')}>Neural</a></li>
                                <li><a href='#' onClick={handleFilterKeyChange('pharmacological')}>Pharmacological</a></li>
                                <li><a href='#' onClick={handleFilterKeyChange('crossscale')}>Cross-Scale</a></li>
                            </ul>
                        </div>
                        <div className="portfolio_list">
                            <div className="filter-container">
                                {figures.map((fig, i) => (
                                    <div key={i} className={`filter-item ${fig.category} fadeInUp`}>
                                        <div className="list_inner">
                                            <div className="image" onClick={() => toggleModal(fig)} style={{cursor: 'pointer'}}>
                                                <img src="img/thumbs/1-1.jpg" alt="" />
                                                <div className="main" data-img-url={fig.img} style={{backgroundImage: `url(${fig.img})`, backgroundSize: 'cover', backgroundPosition: 'center'}}></div>
                                                <span className="icon"><i className="icon-doc-text-inv"></i></span>
                                                <div className="details">
                                                    <h3>{fig.title}</h3>
                                                    <span>{fig.category.charAt(0).toUpperCase() + fig.category.slice(1)}</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {/* <!-- /FIGURES --> */}

            <Modal
                isOpen={isOpen4}
                onRequestClose={closeModal}
                contentLabel="Figure detail"
                className="mymodal"
                overlayClassName="myoverlay"
                closeTimeoutMS={300}
                openTimeoutMS={300}
            >
                <div className="cavani_tm_modalbox opened">
                    <div className="box_inner">
                        <div className="close" onClick={closeModal}>
                            <a href="#">
                                <i className="icon-cancel" />
                            </a>
                        </div>
                        <div className="description_wrap">
                            <div className="popup_details">
                                <div className="top_image">
                                    <img src="img/thumbs/4-2.jpg" alt="" />
                                    <div className="main" style={{ backgroundImage: `url(${modalContent.img})`, backgroundSize: 'cover', backgroundPosition: 'center' }} />
                                </div>
                                <div className="portfolio_main_title">
                                    <h3>{modalContent.title}</h3>
                                    <span>{modalContent.category}</span>
                                </div>
                                <div className="main_details">
                                    <div className="textbox">
                                        <p>{modalContent.caption}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </Modal>

        </>
    )

}
