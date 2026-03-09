import React from 'react'
import ProgressBar from '../progressBar';
import { CircularProgressbar } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import { Swiper, SwiperSlide } from 'swiper/react';
import 'swiper/css';

const circleProgressData = [
    {language: 'Molecular', progress: 100 },
    {language: 'Neural', progress: 100 },
    {language: 'Pharmacological', progress: 100 },
];

const progressBarData = [
    { bgcolor: "#2ecc71", completed: 99, title: 'PLV-Kuramoto R (r=0.988)' },
    { bgcolor: "#3498db", completed: 90, title: 'd_cat vs Efficiency (r=0.902)' },
    { bgcolor: "#9b59b6", completed: 99, title: 'H+ Frequency Ratio (0.99)' },
];

const testimonials = [
    {
        desc: "S_osc = S_cat = S_part = k_B M ln(n) — A single equation unifies oscillatory, categorical, and partition descriptions of biological information filtering.",
        img: "img/figures/panel_4_crossscale.png",
        info1: "Triple Equivalence",
        info2: "Core Result"
    },
    {
        desc: "Zero thermodynamic work (W=0) achieved through topological filtering: the categorical aperture selects information by geometric constraint, not energetic expenditure.",
        img: "img/figures/panel_1_molecular.png",
        info1: "Zero-Work Theorem",
        info2: "Fundamental Principle"
    },
    {
        desc: "PLV correlations of r=0.988 with Kuramoto order parameter across frequency bands validate the framework's predictions for neural phase-locking mechanisms.",
        img: "img/figures/panel_2_neural.png",
        info1: "Neural Validation",
        info2: "EEG Evidence"
    },
]

export default function AboutDefault({ActiveIndex}) {
    return (
        <>
            {/* <!-- FRAMEWORK --> */}
            <div className={ActiveIndex === 1 ? "cavani_tm_section active animated fadeInUp" : "cavani_tm_section active hidden animated"} id="about_">
                <div className="section_inner">
                    <div className="cavani_tm_about">
                        <div className="biography">
                            <div className="cavani_tm_title">
                                <span>The Framework</span>
                            </div>
                            <div className="wrapper">
                                <div className="left">
                                    <p><strong>Categorical Apertures</strong> provide a unified topological framework for understanding how biological systems filter information across scales — from molecular enzymes to neural consciousness.</p>
                                    <p>By mapping information catalysis onto S-entropy coordinates (S_knowledge, S_time, S_entropy), we demonstrate that a single mathematical structure governs filtering at the molecular, neural, and pharmacological scales, spanning 13 orders of magnitude.</p>
                                </div>
                                <div className="right">
                                    <ul>
                                        <li><span className="first">Scale Span:</span><span className="second">13 orders of magnitude</span></li>
                                        <li><span className="first">Regimes:</span><span className="second">5 operational regimes</span></li>
                                        <li><span className="first">Predictions:</span><span className="second">All confirmed (r &gt; 0.90)</span></li>
                                        <li><span className="first">Axioms:</span><span className="second">3 (Closure, Composition, Identity)</span></li>
                                        <li><span className="first">Author:</span><span className="second">Kundai F. Sachikonye</span></li>
                                        <li><span className="first">Affiliation:</span><span className="second"><a href="#">Technical University of Munich</a></span></li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div className="services">
                            <div className="wrapper">
                                <div className="service_list">
                                    <div className="cavani_tm_title">
                                        <span>Core Concepts</span>
                                    </div>
                                    <div className="list">
                                        <ul>
                                            <li>Categorical Apertures (W=0 filtering)</li>
                                            <li>Triple Equivalence Theorem</li>
                                            <li>Variance Minimization Principle</li>
                                            <li>Partition Coordinates</li>
                                            <li>Oscillatory Hole Computing</li>
                                        </ul>
                                    </div>
                                </div>
                                <div className="service_list">
                                    <div className="cavani_tm_title">
                                        <span>Applications</span>
                                    </div>
                                    <div className="list">
                                        <ul>
                                            <li>Depression Thermodynamics</li>
                                            <li>Enzyme Catalysis Geometry</li>
                                            <li>Drug Action Classification</li>
                                            <li>Ion Channel Selectivity</li>
                                            <li>Sleep Stage Transitions</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="skills">
                            <div className="wrapper">
                                <div className="programming">
                                    <div className="cavani_tm_title">
                                        <span>Validation Correlations</span>
                                    </div>
                                    <div className="cavani_progress">
                                        {progressBarData.map((item, idx) => (
                                            <ProgressBar key={idx} bgcolor={item.bgcolor} completed={item.completed} title={item.title} />
                                        ))}
                                    </div>
                                </div>
                                <div className="language">
                                    <div className="cavani_tm_title">
                                        <span>Scale Coverage</span>
                                    </div>
                                    <div className="circular_progress_bar">
                                        <div className='circle_holder'>
                                            {circleProgressData.map((item, idx) => (
                                                <div key={idx}>
                                                    <div className="list_inner">
                                                        <CircularProgressbar
                                                            value={item.progress}
                                                            text={`${item.progress}%`}
                                                            strokeWidth={3}
                                                            stroke='#2ecc71'
                                                            Language={item.language}
                                                            className={"list_inner"}
                                                        />
                                                        <div className="title"><span>{item.language}</span></div>
                                                    </div>
                                                </div>
                                            ))}

                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="resume">
                            <div className="wrapper">
                                <div className="education">
                                    <div className="cavani_tm_title">
                                        <span>Theoretical Foundations</span>
                                    </div>
                                    <div className="list">
                                        <div className="univ">
                                            <ul>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>Axiom 1</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Closure Under Composition</h3>
                                                            <span>Biological information catalysts compose to form new catalysts within the same category</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>Axiom 2</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Associative Composition</h3>
                                                            <span>Sequential information filtering operations are associative: (f . g) . h = f . (g . h)</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>Axiom 3</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Identity Morphisms</h3>
                                                            <span>Each scale possesses identity filtering operations preserving information structure</span>
                                                        </div>
                                                    </div>
                                                </li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                                <div className="experience">
                                    <div className="cavani_tm_title">
                                        <span>Validation Pipeline</span>
                                    </div>
                                    <div className="list">
                                        <div className="univ">
                                            <ul>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>Layer 1</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Molecular Scale</h3>
                                                            <span>KEGG enzyme kinetics, ETC complex analysis, H+ frequency validation</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>Layer 2</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Neural Scale</h3>
                                                            <span>EEG phase-locking, Kuramoto synchronization, depression regime analysis</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>Layer 3</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Pharmacological Scale</h3>
                                                            <span>ChEMBL drug binding, antidepressant response rates, selectivity geometry</span>
                                                        </div>
                                                    </div>
                                                </li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="partners">
                            <div className="cavani_tm_title">
                                <span>Data Sources</span>
                            </div>
                            <div className="list">
                                <ul>
                                    <li>
                                        <div className="list_inner">
                                            <span style={{fontSize: '14px', fontWeight: 'bold', color: '#7d7789'}}>KEGG</span>
                                        </div>
                                    </li>
                                    <li>
                                        <div className="list_inner">
                                            <span style={{fontSize: '14px', fontWeight: 'bold', color: '#7d7789'}}>ChEMBL</span>
                                        </div>
                                    </li>
                                    <li>
                                        <div className="list_inner">
                                            <span style={{fontSize: '14px', fontWeight: 'bold', color: '#7d7789'}}>OpenNeuro</span>
                                        </div>
                                    </li>
                                    <li>
                                        <div className="list_inner">
                                            <span style={{fontSize: '14px', fontWeight: 'bold', color: '#7d7789'}}>PhysioNet</span>
                                        </div>
                                    </li>
                                    <li>
                                        <div className="list_inner">
                                            <span style={{fontSize: '14px', fontWeight: 'bold', color: '#7d7789'}}>UniProt</span>
                                        </div>
                                    </li>
                                    <li>
                                        <div className="list_inner">
                                            <span style={{fontSize: '14px', fontWeight: 'bold', color: '#7d7789'}}>PDB</span>
                                        </div>
                                    </li>
                                </ul>
                            </div>
                        </div>
                        <div className="testimonials">
                            <div className="cavani_tm_title">
                                <span>Key Results</span>
                            </div>
                            <div className="list">
                                <ul className="">
                                    <li>
                                        <Swiper
                                            slidesPerView={1}
                                            spaceBetween={30}
                                            loop={true}
                                            className="custom-class"
                                            breakpoints={{
                                                768: {
                                                    slidesPerView: 2,
                                                }
                                            }}
                                        >
                                            {testimonials.map((item, i) => (
                                                <SwiperSlide key={i}>
                                                    <div className="list_inner">
                                                        <div className="text">
                                                            <i className="icon-quote-left" />
                                                            <p>{item.desc}</p>
                                                        </div>
                                                        <div className="details">
                                                            <div className="image">
                                                                <div className="main" data-img-url={item.img} />
                                                            </div>
                                                            <div className="info">
                                                                <h3>{item.info1}</h3>
                                                                <span>{item.info2}</span>
                                                            </div>
                                                        </div>
                                                    </div>

                                                </SwiperSlide>
                                            ))}
                                        </Swiper>
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {/* <!-- FRAMEWORK --> */}
        </>
    )
}
