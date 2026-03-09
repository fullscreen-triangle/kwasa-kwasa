import React, { useState } from 'react'
import { dataImage } from '../../plugin/plugin'
import Modal from 'react-modal';
import { CloseButton } from '../../plugin/svg';


export default function NewsGlitch({ ActiveIndex }) {

    const [isOpen4, setIsOpen4] = useState(false);
    const [modalContent, setModalContent] = useState({});

    function toggleModalFour() {
        setIsOpen4(!isOpen4);
    }
    const news = [
        {
            img: "img/news/1.jpg",
            tag: "Branding",
            date: "August 9, 2021",
            title: "Building brands through customer service one",
            text1: "Cavani is a leading web design agency with an award-winning design team that creates innovative, effective websites that capture your brand, improve your conversion rates, and maximize your revenue to help grow your business and achieve your goals.",
            text2: "In today&#39;s digital world, your website is the first interaction consumers have with your business. That&#39;s why almost 95 percent of a user&#39;s first impression relates to web design. It&#39;s also why web design services can have an immense impact on your company&#39;s bottom line.",
            text3: "That&#39;s why more companies are not only reevaluating their website&#39;s design but also partnering with Kura, the web design agency that&#39;s driven more than $2.4 billion in revenue for its clients. With over 50 web design awards under our belt, we&#39;re confident we can design a custom website that drives sales for your unique business."
        },
        {
            img: "img/news/2.jpg",
            tag: "Branding",
            date: "August 9, 2021",
            title: "Building brands through customer service two",
            text1: "Cavani is a leading web design agency with an award-winning design team that creates innovative, effective websites that capture your brand, improve your conversion rates, and maximize your revenue to help grow your business and achieve your goals.",
            text2: "In today&#39;s digital world, your website is the first interaction consumers have with your business. That&#39;s why almost 95 percent of a user&#39;s first impression relates to web design. It&#39;s also why web design services can have an immense impact on your company&#39;s bottom line.",
            text3: "That&#39;s why more companies are not only reevaluating their website&#39;s design but also partnering with Kura, the web design agency that&#39;s driven more than $2.4 billion in revenue for its clients. With over 50 web design awards under our belt, we&#39;re confident we can design a custom website that drives sales for your unique business."
        },
        {
            img: "img/news/3.jpg",
            tag: "Branding",
            date: "August 9, 2021",
            title: "Building brands through customer service three",
            text1: "Cavani is a leading web design agency with an award-winning design team that creates innovative, effective websites that capture your brand, improve your conversion rates, and maximize your revenue to help grow your business and achieve your goals.",
            text2: "In today&#39;s digital world, your website is the first interaction consumers have with your business. That&#39;s why almost 95 percent of a user&#39;s first impression relates to web design. It&#39;s also why web design services can have an immense impact on your company&#39;s bottom line.",
            text3: "That&#39;s why more companies are not only reevaluating their website&#39;s design but also partnering with Kura, the web design agency that&#39;s driven more than $2.4 billion in revenue for its clients. With over 50 web design awards under our belt, we&#39;re confident we can design a custom website that drives sales for your unique business."
        },
        {
            img: "img/news/4.jpg",
            tag: "Branding",
            date: "August 9, 2021",
            title: "Building brands through customer service four",
            text1: "Cavani is a leading web design agency with an award-winning design team that creates innovative, effective websites that capture your brand, improve your conversion rates, and maximize your revenue to help grow your business and achieve your goals.",
            text2: "In today&#39;s digital world, your website is the first interaction consumers have with your business. That&#39;s why almost 95 percent of a user&#39;s first impression relates to web design. It&#39;s also why web design services can have an immense impact on your company&#39;s bottom line.",
            text3: "That&#39;s why more companies are not only reevaluating their website&#39;s design but also partnering with Kura, the web design agency that&#39;s driven more than $2.4 billion in revenue for its clients. With over 50 web design awards under our belt, we&#39;re confident we can design a custom website that drives sales for your unique business."
        },
        {
            img: "img/news/5.jpg",
            tag: "Branding",
            date: "August 9, 2021",
            title: "Building brands through customer service five",
            text1: "Cavani is a leading web design agency with an award-winning design team that creates innovative, effective websites that capture your brand, improve your conversion rates, and maximize your revenue to help grow your business and achieve your goals.",
            text2: "In today&#39;s digital world, your website is the first interaction consumers have with your business. That&#39;s why almost 95 percent of a user&#39;s first impression relates to web design. It&#39;s also why web design services can have an immense impact on your company&#39;s bottom line.",
            text3: "That&#39;s why more companies are not only reevaluating their website&#39;s design but also partnering with Kura, the web design agency that&#39;s driven more than $2.4 billion in revenue for its clients. With over 50 web design awards under our belt, we&#39;re confident we can design a custom website that drives sales for your unique business."
        },
        {
            img: "img/news/6.jpg",
            tag: "Branding",
            date: "August 9, 2021",
            title: "Building brands through customer service six",
            text1: "Cavani is a leading web design agency with an award-winning design team that creates innovative, effective websites that capture your brand, improve your conversion rates, and maximize your revenue to help grow your business and achieve your goals.",
            text2: "In today&#39;s digital world, your website is the first interaction consumers have with your business. That&#39;s why almost 95 percent of a user&#39;s first impression relates to web design. It&#39;s also why web design services can have an immense impact on your company&#39;s bottom line.",
            text3: "That&#39;s why more companies are not only reevaluating their website&#39;s design but also partnering with Kura, the web design agency that&#39;s driven more than $2.4 billion in revenue for its clients. With over 50 web design awards under our belt, we&#39;re confident we can design a custom website that drives sales for your unique business."
        }
    ]
    return (
        <>
            {/* <!-- NEWS --> */}
            <div className={ActiveIndex === 3 ? "cavani_tm_section active animated rollIn" : "cavani_tm_section hidden animated rollOut"} id="news_">

                <div className="section_inner">
                    <div className="cavani_tm_news">
                        <div className="cavani_tm_title">
                            <span>Latest News</span>
                        </div>
                        <div className="news_list">
                            <ul>
                                {news.map((item, i) => (
                                    <li key={i}>
                                        <div className="list_inner">
                                            <div className="image" onClick={toggleModalFour}>
                                                <img src="img/thumbs/4-3.jpg" alt="" />
                                                <div className="main" data-img-url={item.img}></div>
                                                <a className="cavani_tm_full_link" href="#" onClick={() => setModalContent(item)}></a>
                                            </div>
                                            <div className="details">
                                                <div className="metabox">
                                                    <div className="info">
                                                        <span className="category"><a href="#">{item.tag}</a></span>
                                                        <span className="date">{item.date}</span>
                                                    </div>
                                                    <div className="title" onClick={toggleModalFour}>
                                                        <h3><a href="#">{item.title}</a></h3>
                                                    </div>
                                                </div>
                                            </div>

                                            <div className="news_hidden_details">
                                                <div className="news_popup_informations">
                                                    <div className="text">
                                                        <p> {item.text1}</p>
                                                        <p> {item.text2}</p>
                                                        <p> {item.text3}</p>
                                                    </div>
                                                </div>
                                            </div>

                                        </div>
                                    </li>))}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            {/* <!-- NEWS --> */}

            {modalContent && (
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
                            <div className="close" onClick={toggleModalFour} >
                                <a href="#"><i className="icon-cancel"></i></a>
                            </div>
                            <div className="description_wrap">
                                <div className="news_popup_informations">
                                    <div className="image">
                                        <img src="img/thumbs/4-2.jpg" alt="" />
                                        <div className="main" data-img-url="img/news/1.jpg" style={{ backgroundImage: `url(${modalContent.img})` }} />
                                    </div>
                                    <div className="details">
                                        <span>{modalContent.tag}</span>
                                        <h3>{modalContent.title}</h3>
                                    </div>
                                    <div className="text">
                                        {modalContent.text1}
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