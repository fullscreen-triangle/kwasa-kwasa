import React from 'react'
import Link from 'next/link'

export default function Header({handleOnClick, ActiveIndex}) {
    
    return (
        <>
            {/* HEADER */}
            <div className="cavani_tm_header">
                <div className="logo">
                    <a href="#"><img src="img/logo/dark.png" alt="" /></a>
                </div>
                <div className="menu">
                    <ul className="transition_link">
                        <li onClick={() => handleOnClick(0)}><a className={ActiveIndex === 0 ? "active" : ""}>Home</a></li>
                        <li onClick={() => handleOnClick(1)}><a className={ActiveIndex === 1 ? "active" : ""}>Framework</a></li>
                        <li onClick={() => handleOnClick(2)}><a className={ActiveIndex === 2 ? "active" : ""}>Figures</a></li>
                        <li onClick={() => handleOnClick(7)}><a className={ActiveIndex === 7 ? "active" : ""}>Validation</a></li>
                        <li onClick={() => handleOnClick(3)}><a className={ActiveIndex === 3 ? "active" : ""}>Publications</a></li>
                        <li onClick={() => handleOnClick(4)}><a className={ActiveIndex === 4 ? "active" : ""}>Collaborate</a></li>
                    </ul>
                    {/* <span className="ccc" /> */}
                </div>
            </div>
            {/* /HEADER */}

        </>
    )
}
