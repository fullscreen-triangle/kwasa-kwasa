import {
    LightgalleryProvider,
    LightgalleryItem
} from "react-lightgallery";

import "lightgallery.js/dist/css/lightgallery.css";

export const PhotoItem = ({ image, thumb }) => (

    <LightgalleryItem src={image} thumb={thumb}>
        <div className="filter-item image">
            <div className="list_inner">
                <div className="image">
                    <img src="img/thumbs/1-1.jpg" alt="" />
                    <a className="cavani_tm_full_link zoom" >
                        <div className="main" style={{ backgroundImage: `url(${image})` }}></div>
                        <span className="icon"><i className="icon-resize-full-alt"></i></span>
                        <div className="details">
                            <h3>Alla Gorova</h3>
                            <span>Image</span>
                        </div>
                    </a>
                </div>
            </div>
        </div>
    </LightgalleryItem>

);

{/* <li className="image" key={image.id}>
    <div className="list_inner">
        <div className="image">
            <img src="img/thumbs/1-1.jpg" alt="" />
            <a className="cavani_tm_full_link zoom" >
                <div className="main" style={{ backgroundImage: `url(${image.image})` }}></div>
                <span className="icon"><i className="icon-resize-full-alt"></i></span>
                <div className="details">
                    <h3>Alla Gorova</h3>
                    <span>Image</span>
                </div>
            </a>
        </div>
    </div>
</li> */}