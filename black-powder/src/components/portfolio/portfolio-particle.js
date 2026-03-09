import React, { useState } from 'react'
import Image from 'next/image';
import { dataImage } from '../../plugin/plugin'
import { LightgalleryProvider } from 'react-lightgallery';
import { PhotoItem } from '../../plugin/lightGalleryOptions'
import dynamic from 'next/dynamic';
import Modal from 'react-modal';
import { CloseButton } from '../../plugin/svg';

export default function PortfolioDefault({ ActiveIndex }) {

    const [activeTab, setActiveTab] = useState(1);

    const handleOnClick = (index) => {
        setActiveTab(index); // remove the curly braces
    };

    // for popup video
    const ModalVideo = dynamic(
        () => {
            return import("react-modal-video");
        },
        { ssr: false }
    );


    const [isOpen, setIsOpen] = useState(false);
    const [isOpen2, setIsOpen2] = useState(false);
    const [isOpen3, setIsOpen3] = useState(false);

    function toggleModalThree() {
        setIsOpen3(!isOpen3);
    }


    // image modal popup
    const [isOpen4, setIsOpen4] = useState(false);
    const [isOpen5, setIsOpen5] = useState(false);

    function toggleModalFour() {
        setIsOpen4(!isOpen4);
    }
    function toggleModalFive() {
        setIsOpen5(!isOpen5);
    }

    const imagesCollection = [
        [
            "img/portfolio/4.jpg",
        ],
        [
            "img/portfolio/5.jpg",
        ]
    ];

    return (
        <>
            {/* <!-- PORTFOLIO --> */}

            <div className={ActiveIndex === 2 ? "cavani_tm_section active animated flipInX" : "cavani_tm_section hidden animated flipOutX"} id="portfolio_">
                <div className="section_inner">
                    <div className="cavani_tm_portfolio">
                        <div className="cavani_tm_title">
                            <span>Creative Portfolio</span>
                        </div>


                        <div className="portfolio_filter">
                            <ul>
                                <li><a onClick={() => handleOnClick(1)} href="#" className="current" data-filter="*">All</a></li>
                                <li><a onClick={() => handleOnClick(2)} href="#" data-filter=".vimeo">Vimeo</a></li>
                                <li><a onClick={() => handleOnClick(3)} href="#" data-filter=".youtube">Youtube</a></li>
                                <li><a onClick={() => handleOnClick(4)} href="#" data-filter=".soundcloud">Soundcloud</a></li>
                                <li><a onClick={() => handleOnClick(6)} href="#" data-filter=".detail">Detail</a></li>
                                <li><a onClick={() => handleOnClick(5)} href="#" data-filter=".image">Image</a></li>
                            </ul>
                        </div>
                        <div className="portfolio_list">

                            <ul className={activeTab === 1 ? "gallery_zoom" : "gallery_zoom d-none"}>
                                <li className="vimeo fadeInUp">
                                    <div className="list_inner">
                                        <div className="image">
                                            <img src="img/thumbs/1-1.jpg" alt="" />
                                            <div className="main" data-img-url="img/portfolio/2.jpg" onClick={() => setIsOpen2(true)}></div>
                                            <span className="icon"><i className="icon-vimeo-1"></i></span>
                                            <div className="details">
                                                <h3>Kelly Hookin</h3>
                                                <span>Vimeo</span>
                                            </div>
                                        </div>
                                    </div>
                                </li>
                                <li className="youtube">
                                    <div className="list_inner">
                                        <div className="image">
                                            <img src="img/thumbs/1-1.jpg" alt="" />
                                            <div className="main" data-img-url="img/portfolio/1.jpg" onClick={() => setIsOpen(true)}></div>
                                            <span className="icon"><i className="icon-youtube-play"></i></span>
                                            <div className="details">
                                                <h3>Ave Brook</h3>
                                                <span>Youtube</span>
                                            </div>
                                            {/* <a className="cavani_tm_full_link popup-youtube" href="https://www.youtube.com/watch?v=7e90gBu4pas"></a> */}
                                        </div>
                                    </div>
                                </li>
                                <li className="soundcloud">
                                    <div className="list_inner">
                                        <div className="image">
                                            <img src="img/thumbs/1-1.jpg" alt="" />
                                            <div className="main" data-img-url="img/portfolio/3.jpg" onClick={toggleModalThree}></div>
                                            <span className="icon"><i className="icon-soundcloud-1"></i></span>
                                            <div className="details">
                                                <h3>Ashley Flores</h3>
                                                <span>Soundcloud</span>
                                            </div>
                                            {/* <a className="cavani_tm_full_link soundcloude_link mfp-iframe audio" href="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/471954807&color=%23ff5500&auto_play=true&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></a> */}
                                        </div>
                                    </div>
                                </li>
                                <li className="detail">
                                    <div className="list_inner">
                                        <div className="image">
                                            <img src="img/thumbs/1-1.jpg" alt="" />
                                            <div className="main" data-img-url="img/portfolio/6.jpg" onClick={toggleModalFour}></div>
                                            <span className="icon"><i className="icon-doc-text-inv"></i></span>
                                            <div className="details">
                                                <h3>Alice Moon</h3>
                                                <span>Detail</span>
                                            </div>
                                            {/* <a className="cavani_tm_full_link portfolio_popup" href="#"></a> */}
                                        </div>



                                    </div>
                                </li>

                                {/* Photo Gallery Started */}
                                <LightgalleryProvider>

                                    {imagesCollection.map((p, idx) => (
                                        <PhotoItem key={idx} image={p[0]} thumb={p[1]} />
                                    ))}

                                </LightgalleryProvider>
                                {/* Photo Gallery End */}
                            </ul>

                            <ul className={activeTab === 2 ? "gallery_zoom" : "gallery_zoom d-none"} >
                                <li className="vimeo">
                                    <div className="list_inner">
                                        <div className="image">
                                            <img src="img/thumbs/1-1.jpg" alt="" />
                                            <div className="main" data-img-url="img/portfolio/2.jpg" onClick={() => setIsOpen2(true)}></div>
                                            <span className="icon"><i className="icon-vimeo-1"></i></span>
                                            <div className="details">
                                                <h3>Kelly Hookin</h3>
                                                <span>Vimeo</span>
                                            </div>
                                        </div>
                                    </div>
                                </li>
                            </ul>


                            <ul className={activeTab === 3 ? "gallery_zoom" : "gallery_zoom d-none"}>
                                <li className="youtube">
                                    <div className="list_inner">
                                        <div className="image">
                                            <img src="img/thumbs/1-1.jpg" alt="" />
                                            <div className="main" data-img-url="img/portfolio/1.jpg" onClick={() => setIsOpen(true)}></div>
                                            <span className="icon"><i className="icon-youtube-play"></i></span>
                                            <div className="details">
                                                <h3>Ave Brook</h3>
                                                <span>Youtube</span>
                                            </div>
                                            {/* <a className="cavani_tm_full_link popup-youtube" href="https://www.youtube.com/watch?v=7e90gBu4pas"></a> */}
                                        </div>
                                    </div>
                                </li>
                            </ul>


                            <ul className={activeTab === 4 ? "gallery_zoom" : "gallery_zoom d-none"}>
                                <li className="soundcloud">
                                    <div className="list_inner">
                                        <div className="image">
                                            <img src="img/thumbs/1-1.jpg" alt="" />
                                            <div className="main" data-img-url="img/portfolio/3.jpg" onClick={toggleModalThree}></div>
                                            <span className="icon"><i className="icon-soundcloud-1"></i></span>
                                            <div className="details">
                                                <h3>Ashley Flores</h3>
                                                <span>Soundcloud</span>
                                            </div>
                                            {/* <a className="cavani_tm_full_link soundcloude_link mfp-iframe audio" href="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/471954807&color=%23ff5500&auto_play=true&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></a> */}
                                        </div>
                                    </div>
                                </li>
                            </ul>


                            <ul className={activeTab === 5 ? "gallery_zoom" : "gallery_zoom d-none"}>
                                {/* Photo Gallery Started */}
                                <LightgalleryProvider>

                                    {imagesCollection.map((p, idx) => (
                                        <PhotoItem key={idx} image={p[0]} thumb={p[1]} />
                                    ))}

                                </LightgalleryProvider>
                                {/* Photo Gallery End */}
                            </ul>


                            <ul className={activeTab === 6 ? "gallery_zoom" : "gallery_zoom d-none"}>
                                <li className="detail">
                                    <div className="list_inner">
                                        <div className="image">
                                            <img src="img/thumbs/1-1.jpg" alt="" />
                                            <div className="main" data-img-url="img/portfolio/6.jpg" onClick={toggleModalFour}></div>
                                            <span className="icon"><i className="icon-doc-text-inv"></i></span>
                                            <div className="details">
                                                <h3>Alice Moon</h3>
                                                <span>Detail</span>
                                            </div>
                                            {/* <a className="cavani_tm_full_link portfolio_popup" href="#"></a> */}
                                        </div>



                                    </div>
                                </li>
                            </ul>


                        </div>
                    </div>
                </div>
            </div>
            {/* <!-- /PORTFOLIO --> */}

            <ModalVideo
                channel="youtube"
                autoplay={true}
                isOpen={isOpen}
                videoId="7e90gBu4pas"
                animationSpeed={300}
                onClose={() => setIsOpen(false)}
                modalVideoClose="mfp-close"
            />
            <ModalVideo
                channel="vimeo"
                autoplay
                isOpen={isOpen2}
                videoId="337293658"
                animationSpeed={300}
                onClose={() => setIsOpen2(false)}
                modalVideoClose="mfp-close"
            />
            {/* <ModalVideo
                channel=""
                autoplay
                isOpen={isOpen3}
                videoId="https://soundcloud.com/bestnotesound/upbeat-corporate-uplifting-motivational-download-free-music-royalty-free-music-background?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing"
                onClose={() => setOpen3(false)}
            /> */}
            <Modal
                isOpen={isOpen3}
                onRequestClose={toggleModalThree}
                contentLabel="My dialog"
                className="mymodal"
                overlayClassName="myoverlay"
                closeTimeoutMS={300}
                openTimeoutMS={300}
            >

                <div onClick={toggleModalThree} className="modal-video mfp-wrap mfp-gallery mfp-close-btn-in mfp-auto-cursor mfp-ready" tabIndex={-1} style={{ overflow: 'hidden auto' }}>
                    <div className="mfp-container mfp-s-ready mfp-iframe-holder">
                        <div className="mfp-content">
                            <div className="mfp-iframe-scaler">
                                <button onClick={toggleModalThree} title="Close (Esc)" type="button" className="mfp-close">×</button>
                                <iframe className="mfp-iframe" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/471954807&color=%23ff5500&auto_play=true&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true" frameBorder={0} allowFullScreen />
                            </div>
                        </div>
                        <div className="mfp-preloader">
                            Loading...
                        </div>
                    </div>
                </div>
            </Modal>
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
                                <i className="icon-cancel" />
                            </a>
                        </div>
                        <div className="description_wrap">
                            <div className="popup_details">
                                <div className="top_image">
                                    <img src="img/thumbs/4-2.jpg" alt="" />
                                    <div className="main" data-img-url="img/portfolio/6.jpg" style={{ backgroundImage: "url(./img/portfolio/6.jpg" }} />
                                </div>
                                <div class="portfolio_main_title">
                                    <h3>Alice Moon</h3>
                                    <span>Detail</span>
                                    <div></div>
                                </div>
                                <div className="main_details">
                                    <div className="textbox">
                                        <p>We live in a world where we need to move quickly and iterate on our ideas as flexibly as possible. Building mockups strikes the ideal balance ease of modification. Building mockups strikes the ideal balance ease of modification.</p>
                                        <p>Mockups are useful both for the creative phase of the project - for instance when you&#39;re trying to figure out your user flows or the proper visual hierarchy - and the production phase when they phase when they will represent the target product. Building mockups strikes the ideal balance ease of modification.</p>
                                    </div>
                                    <div className="detailbox">
                                        <ul>
                                            <li>
                                                <span className="first">Client</span>
                                                <span>Alvaro Morata</span>
                                            </li>
                                            <li>
                                                <span className="first">Category</span>
                                                <span><a href="#">Detail</a></span>
                                            </li>
                                            <li>
                                                <span className="first">Date</span>
                                                <span>March 07, 2021</span>
                                            </li>
                                            <li>
                                                <span className="first">Share</span>
                                                <ul className="share">
                                                    <li><a href="#"><img className="svg" src="./img/svg/social/facebook.svg" alt="" /></a></li>
                                                    <li><a href="#"><img className="svg" src="./img/svg/social/twitter.svg" alt="" /></a></li>
                                                    <li><a href="#"><img className="svg" src="./img/svg/social/instagram.svg" alt="" /></a>
                                                    </li>
                                                </ul>
                                            </li>
                                        </ul>
                                    </div>
                                </div>
                                <div className="additional_images">
                                    <ul>
                                        <li>
                                            <div className="list_inner">
                                                <div className="my_image">
                                                    <img src="./img/thumbs/4-2.jpg" alt="" />
                                                    <div className="main" style={{ backgroundImage: "url(./img/portfolio/1.jpg" }}></div>

                                                </div>
                                            </div>
                                        </li>
                                        <li>
                                            <div className="list_inner">
                                                <div className="my_image">
                                                    <img src="./img/thumbs/4-2.jpg" alt="" />
                                                    <div className="main" style={{ backgroundImage: "url(./img/portfolio/2.jpg" }}></div>
                                                </div>
                                            </div>
                                        </li>
                                        <li>
                                            <div className="list_inner">
                                                <div className="my_image">
                                                    <img src="./img/thumbs/4-2.jpg" alt="" />
                                                    <div className="main" style={{ backgroundImage: "url(./img/portfolio/3.jpg" }}></div>
                                                </div>
                                            </div>
                                        </li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </Modal>

        </>
    )

}
