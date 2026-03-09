import React, { useState } from 'react'
import Modal from 'react-modal';
import {SVG_Custom1, SVG_Custom2, SVG_Custom3 } from '../../plugin/svg';
export default function Service({ ActiveIndex }) {

    const [isOpen7, setIsOpen7] = useState(false);
    const [modalContent, setModalContent] = useState({});

    function toggleModalFour() {
        setIsOpen7(!isOpen7);
    }
    const service = [
        {
            img: "img/figures/panel_1_molecular.png",
            svg: <SVG_Custom1 />,
            text: "KEGG enzyme kinetics, ETC complex flux analysis, and H+ frequency validation across 6 enzyme classes with r=0.902 correlation.",
            title: "Molecular Validation",
            text1: "The molecular validation layer tests the categorical aperture framework against established enzyme kinetics data from the KEGG database. We analyze 6 major enzyme classes (oxidoreductases, transferases, hydrolases, lyases, isomerases, ligases) and demonstrate that categorical distance d_cat correlates with catalytic efficiency at r=0.902.",
            text2: "The electron transport chain (ETC) analysis reveals that proton flux through Complexes I-V follows the predicted aperture geometry. The H+ oscillation frequency of ~40 THz (10^13 Hz) emerges naturally from the framework's partition coordinates, matching experimental observations.",
            text3: "Key results: Cytochrome c oxidase selectivity ratio 0.99, ETC complex flux predictions within 5% of experimental values, and S-entropy coordinate clustering correctly separates enzyme functional classes in 3D space."
        },
        {
            img: "img/figures/panel_2_neural.png",
            svg: <SVG_Custom2 />,
            text: "EEG phase-locking analysis showing PLV-Kuramoto correlation r=0.988 across frequency bands, with depression regime classification.",
            title: "Neural Validation",
            text1: "The neural validation layer demonstrates that phase-locking value (PLV) across EEG frequency bands follows Kuramoto synchronization dynamics predicted by the categorical aperture framework. Across theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), and gamma (30-100 Hz) bands, the correlation between PLV and Kuramoto order parameter R reaches r=0.988.",
            text2: "Depression is characterized as a regime transition: healthy subjects occupy the Phase-Locked regime (R>0.7), while depressed subjects show Turbulent regime characteristics (R<0.4). Treatment response maps to regime re-entry, with PLV increasing from 0.32 to 0.77 during successful SSRI treatment.",
            text3: "Key results: HAM-D scores decrease from 24.3 to 8.5, 83% treatment response rate, variance minimization at 40 Hz gamma band confirming the oscillatory hole computing prediction, and 5 operational regimes cleanly separable in the (R, sigma^2, S) phase space."
        },
        {
            img: "img/figures/panel_3_pharma.png",
            svg: <SVG_Custom3 />,
            text: "ChEMBL drug binding profiles mapped to S-entropy coordinates, with selectivity geometry predicting clinical response rates.",
            title: "Pharmacological Validation",
            text1: "The pharmacological validation layer maps antidepressant drug binding profiles from ChEMBL onto S-entropy coordinates. Each drug's multi-receptor binding pattern (5-HT, NET, DAT, sigma-1, NMDA) defines a unique trajectory through the categorical aperture space.",
            text2: "The framework predicts that drugs with lower categorical distance d_cat to the healthy regime boundary will show higher clinical response rates. This prediction is confirmed: SSRIs (d_cat=0.23, response=67%), SNRIs (d_cat=0.31, response=58%), and atypical agents like ketamine (d_cat=0.15, response=71%) all follow the predicted geometry.",
            text3: "Key results: Drug selectivity profiles cluster into 3 distinct aperture classes, binding affinity ratios predict regime transition probabilities, and the cross-scale unification shows identical mathematical structure at molecular (enzyme), neural (EEG), and pharmacological (drug) scales."
        }
    ]
    return (
        <>
            {/* <!-- VALIDATION --> */}
            <div className={ActiveIndex === 7 ? "cavani_tm_section active animated fadeInUp" : "cavani_tm_section hidden animated"} id="news_">
                <div className="section_inner">
                    <div className="cavani_tm_service">
                        <div className="cavani_tm_title">
                            <span>Three-Layer Validation</span>
                        </div>
                        <div className="service_list">
                            <ul>
                                {service.map((item, i) => (
                                    <li key={i}>
                                        <div className="list_inner" onClick={toggleModalFour}>
                                            {item.svg}
                                            <h3 className="title" onClick={toggleModalFour}>{item.title}</h3>
                                            <p className="text">{item.text}</p>
                                            <a className="cavani_tm_full_link" href="#" onClick={() => setModalContent(item)} />
                                            <img className="popup_service_image" src={item.img} alt="" />
                                            <div className="service_hidden_details">
                                                <div className="service_popup_informations">
                                                    <div className="descriptions">
                                                        <p>{item.text1}</p>
                                                        <p>{item.text2}</p>
                                                        <p>{item.text3}</p>
                                                    </div>
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
            {/* <!-- VALIDATION --> */}

            {modalContent && (
                <Modal
                    isOpen={isOpen7}
                    onRequestClose={toggleModalFour}
                    contentLabel="My dialog"
                    className="mymodal"
                    overlayClassName="myoverlay"
                    closeTimeoutMS={300}
                    openTimeoutMS={300}
                >
                    <div className="cavani_tm_modalbox opened">
                        <div className="box_inner">
                            <div className="close" onClick={toggleModalFour} >
                                <a href="#"><i className="icon-cancel"></i></a>
                            </div>
                            <div className="description_wrap">
                                <div className="service_popup_informations">
                                    <div className="image">
                                        <img src="img/thumbs/4-2.jpg" alt="" />
                                        <div className="main" data-img-url={modalContent.img} style={{ backgroundImage: `url(${modalContent.img})` }} />
                                    </div>
                                    <div className="details">
                                        <span>Validation Layer</span>
                                        <h3>{modalContent.title}</h3>
                                    </div>
                                    <div className="descriptions">
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
        </>
    )
}
