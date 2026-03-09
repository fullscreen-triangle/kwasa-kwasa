import React from 'react'

export default function Intro() {
    return (
        <div className="cavani_tm_intro">
            <div className="main_title">
                <img src="img/logo/dark.png" alt="" />
                <p>CV / Resume / Personal / Portfolio Template</p>
            </div>
            <div className="demo_list">
                <ul>
                    <li>
                        <div className="list_inner">
                            <img src="img/intro/1.jpg" alt="" />
                            <h3>Static Demo</h3>
                            <a className="cavani_tm_full_link" href="/" target="_blank" />
                        </div>
                    </li>
                    <li>
                        <div className="list_inner">
                            <img src="img/intro/2.jpg" alt="" />
                            <h3>Glitch Demo</h3>
                            <a className="cavani_tm_full_link" href="/index-glitch" target="_blank" />
                        </div>
                    </li>
                    <li>
                        <div className="list_inner">
                            <img src="img/intro/3.jpg" alt="" />
                            <h3>Water Demo</h3>
                            <a className="cavani_tm_full_link" href="/index-water" target="_blank" />
                        </div>
                    </li>
                    <li>
                        <div className="list_inner">
                            <img src="img/intro/4.jpg" alt="" />
                            <h3>Particle Demo</h3>
                            <a className="cavani_tm_full_link" href="/index-particle" target="_blank" />
                        </div>
                    </li>
                </ul>
            </div>
        </div>

    )
}
