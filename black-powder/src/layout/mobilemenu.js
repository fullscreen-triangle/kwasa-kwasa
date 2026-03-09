import React,{useEffect} from 'react'
import { dataImage } from '../plugin/plugin'

export default function Mobilemenu({isToggled, handleOnClick}) {
  useEffect(() => {
    dataImage();
  });
    return (
        <>

            {/* MOBILE MENU */}
            <div className={!isToggled ? "cavani_tm_mobile_menu" :  "cavani_tm_mobile_menu opened"} >
                <div className="inner">
                    <div className="wrapper">
                        <div className="avatar">
                            <div className="image" data-img-url="img/figures/panel_4_crossscale.png" />
                        </div>
                        <div className="menu_list">
                            <ul className="transition_link">
                                <li onClick={() => handleOnClick(0)}><a href="#home">Home</a></li>
                                <li onClick={() => handleOnClick(1)}><a href="#framework">Framework</a></li>
                                <li onClick={() => handleOnClick(2)}><a href="#figures">Figures</a></li>
                                <li onClick={() => handleOnClick(7)}><a href="#validation">Validation</a></li>
                                <li onClick={() => handleOnClick(3)}><a href="#publications">Publications</a></li>
                                <li onClick={() => handleOnClick(4)}><a href="#collaborate">Collaborate</a></li>
                            </ul>
                        </div>
                        <div className="social">
                            <ul>
                                <li><a href="https://github.com/kundai-sachikonye" target="_blank" rel="noopener noreferrer"><i className="icon-github-circled"></i></a></li>
                                <li><a href="mailto:kundai.sachikonye@tum.de"><i className="icon-mail-3"></i></a></li>
                            </ul>
                        </div>
                        <div className="copyright">
                            <p>Copyright © 2025 Kundai F. Sachikonye</p>
                        </div>
                    </div>
                </div>
            </div>
            {/* /MOBILE MENU */}


        </>
    )
}
