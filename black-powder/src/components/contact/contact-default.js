import React, { useState, useEffect } from 'react'
import MagicCursor from '../../layout/magic-cursor';
import { customCursor } from '../../plugin/plugin';

export default function ContactDefault({ ActiveIndex }) {
    const [trigger, setTrigger] = useState(false);
    useEffect(() => {
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
            {/* <!-- COLLABORATE --> */}
            <div className={ActiveIndex === 4 ? "cavani_tm_section active animated fadeInUp" : "cavani_tm_section hidden animated"} id="contact_">
                <div className="section_inner">
                    <div className="cavani_tm_contact">
                        <div className="cavani_tm_title">
                            <span>Collaborate</span>
                        </div>

                        <div className="short_info">
                            <ul>
                                <li>
                                    <div className="list_inner">
                                        <i className="icon-location"></i>
                                        <span>Technical University of Munich</span>
                                    </div>
                                </li>
                                <li>
                                    <div className="list_inner">
                                        <i className="icon-mail-3"></i>
                                        <span><a href="mailto:kundai.sachikonye@tum.de">kundai.sachikonye@tum.de</a></span>
                                    </div>
                                </li>
                                <li>
                                    <div className="list_inner">
                                        <i className="icon-github-circled"></i>
                                        <span><a href="https://github.com/kundai-sachikonye" target="_blank" rel="noopener noreferrer">GitHub</a></span>
                                    </div>
                                </li>
                            </ul>
                        </div>

                        <div className="funding_interests" style={{marginBottom: '40px'}}>
                            <div className="cavani_tm_title">
                                <span>Funding Interests</span>
                            </div>
                            <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px', marginTop: '20px'}}>
                                <div style={{padding: '20px', border: '1px solid rgba(125,119,137,0.2)', borderRadius: '8px'}}>
                                    <h4 style={{marginBottom: '10px', color: '#2ecc71'}}>Experimental Validation</h4>
                                    <p style={{fontSize: '14px', color: '#7d7789'}}>Laboratory confirmation of predicted H+ oscillation frequencies and enzyme selectivity ratios using spectroscopic methods.</p>
                                </div>
                                <div style={{padding: '20px', border: '1px solid rgba(125,119,137,0.2)', borderRadius: '8px'}}>
                                    <h4 style={{marginBottom: '10px', color: '#3498db'}}>Clinical Translation</h4>
                                    <p style={{fontSize: '14px', color: '#7d7789'}}>EEG-based regime classification for depression treatment optimization using real-time phase-locking analysis.</p>
                                </div>
                                <div style={{padding: '20px', border: '1px solid rgba(125,119,137,0.2)', borderRadius: '8px'}}>
                                    <h4 style={{marginBottom: '10px', color: '#9b59b6'}}>Computational Infrastructure</h4>
                                    <p style={{fontSize: '14px', color: '#7d7789'}}>Development of the Turbulance DSL and kwasa-kwasa runtime for semantic computation at scale.</p>
                                </div>
                            </div>
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
                                            src="https://maps.google.com/maps?q=Technical%20University%20of%20Munich&t=&z=13&ie=UTF8&iwloc=&output=embed"
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {/* <!-- COLLABORATE --> */}
        </>
    )
}
