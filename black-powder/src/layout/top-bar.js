import React, {useState} from 'react'

export default function TopBar({toggleTrueFalse, isToggled}) {
       

    return (
        <>
            <div className="cavani_tm_topbar">
                <div className="topbar_inner">
                    <div className="logo">
                        <a href="#"><img src="img/logo/dark.png" alt="" /></a>
                    </div>
                    <div className="trigger">
                        <div onClick={toggleTrueFalse} className={!isToggled ? "hamburger hamburger--slider" : "hamburger hamburger--slider is-active"}>
                            <div className="hamburger-box">
                                <div className="hamburger-inner"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}
