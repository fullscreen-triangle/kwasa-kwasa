import React, { useState, useEffect } from 'react'
import MagicCursor from '../../layout/magic-cursor';
import { customCursor } from '../../plugin/plugin';

export default function ContactDefault({ ActiveIndex }) {
    const [trigger, setTrigger] = useState(false);
    useEffect(() => {
        // dataImage();
        customCursor();
    });

    const [form, setForm] = useState({ email: "", name: "", msg: "" });
    const [active, setActive] = useState(null);
    const [error, setError] = useState(false);
    const [success, setSuccess] = useState(false);
    const onChange = (e) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    };
    const { email, name, msg } = form;
    const onSubmit = (e) => {
        e.preventDefault();
        if (email && name && msg) {
            setSuccess(true);
            setTimeout(() => {
                setForm({ email: "", name: "", msg: "" });
                setSuccess(false);
            }, 2000);
        } else {
            setError(true);
            setTimeout(() => {
                setError(false);
            }, 2000);
        }
    };
    return (
        <>
            {/* <!-- CONTACT --> */}
            <div className={ActiveIndex === 4 ? "cavani_tm_section active animated flipInX" : "cavani_tm_section hidden animated flipOutX"} id="contact_">
            <div className="section_inner">
                    <div className="cavani_tm_contact">
                        <div className="cavani_tm_title">
                            <span>Get in Touch</span>
                        </div>
                        <div className="short_info">
                            <ul>
                                <li>
                                    <div className="list_inner">
                                        <i className="icon-location"></i>
                                        <span>Ave Street, New York, USA</span>
                                    </div>
                                </li>
                                <li>
                                    <div className="list_inner">
                                        <i className="icon-mail-3"></i>
                                        <span><a href="#">hello@cavani.com</a></span>
                                    </div>
                                </li>
                                <li>
                                    <div className="list_inner">
                                        <i className="icon-mobile"></i>
                                        <span>+77 022 444 05 05</span>
                                    </div>
                                </li>
                            </ul>
                        </div>
                        <div className="form">
                            <div className="left">
                                <div className="fields">
                                    {/* Contact Form */}
                                    <form className="contact_form" onSubmit={(e) => onSubmit(e)}>
                                        <div
                                            className="returnmessage"
                                            data-success="Your message has been received, we will contact you soon."
                                            style={{ display: success ? "block" : "none" }}
                                        >
                                            <span className="contact_success">
                                                Your message has been received, we will contact you soon.
                                            </span>
                                        </div>
                                        <div
                                            className="empty_notice"
                                            style={{ display: error ? "block" : "none" }}
                                        >
                                            <span>Please Fill Required Fields!</span>
                                        </div>
                                        {/* */}

                                        <div className="fields">
                                            <ul>
                                                <li
                                                    className={`input_wrapper ${active === "name" || name ? "active" : ""
                                                        }`}
                                                >
                                                    <input
                                                        onFocus={() => setActive("name")}
                                                        onBlur={() => setActive(null)}
                                                        onChange={(e) => onChange(e)}
                                                        value={name}
                                                        name="name"
                                                        id="name"
                                                        type="text"
                                                        placeholder="Name"
                                                    />
                                                </li>
                                                <li
                                                    className={`input_wrapper ${active === "email" || email ? "active" : ""
                                                        }`}
                                                >
                                                    <input
                                                        onFocus={() => setActive("email")}
                                                        onBlur={() => setActive(null)}
                                                        onChange={(e) => onChange(e)}
                                                        value={email}
                                                        name="email"
                                                        id="email"
                                                        type="email"
                                                        placeholder="Email"
                                                    />
                                                </li>
                                                <li
                                                    className={`last ${active === "message" || msg ? "active" : ""
                                                        }`}
                                                >
                                                    <textarea
                                                        onFocus={() => setActive("message")}
                                                        onBlur={() => setActive(null)}
                                                        name="msg"
                                                        onChange={(e) => onChange(e)}
                                                        value={msg}
                                                        id="message"
                                                        placeholder="Message"
                                                    />
                                                </li>
                                            </ul>
                                            <div className="cavani_tm_button">
                                                <input
                                                    className='a'
                                                    type="submit"
                                                    id="send_message"
                                                    value="Send Message"
                                                />
                                            </div>
                                        </div>
                                    </form>
                                    {/* /Contact Form */}
                                </div>
                            </div>
                            <div className="right">
                                <div className="map_wrap">
                                    <div className="map" id="ieatmaps">
                                        <iframe
                                            height={375}
                                            style={{ width: "100%" }}
                                            id="gmap_canvas"
                                            src="https://maps.google.com/maps?q=2880%20Broadway,%20New%20York&t=&z=13&ie=UTF8&iwloc=&output=embed"
                                        />
                                        <a href="https://www.embedgooglemap.net/blog/divi-discount-code-elegant-themes-coupon" />
                                        <br />
                                    </div>
                                </div>
                                {/* Get your API here https://www.embedgooglemap.net */}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {/* <!-- CONTACT --> */}
        </>
    )
}
