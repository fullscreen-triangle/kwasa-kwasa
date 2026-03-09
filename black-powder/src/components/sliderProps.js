import SwiperCore, {
    Autoplay,
    EffectFade,
    Navigation,
    Pagination,
  } from "swiper";
  SwiperCore.use([Pagination, Navigation, Autoplay, EffectFade]);
  
 
  export const testimonialSliderProps = {
    slidesPerView: 2,
    loop: true,
    spaceBetween: 30,
    speed: 1000,
    autoplay: {
      delay: 2500,
      disableOnInteraction: true,
    },
    navigation: {
      nextEl: ".next",
      prevEl: ".prev",
    },
    breakpoints: {
      480: {
        slidesPerView: 2,
      },
      1400: {
        slidesPerView: 3,
      },
    },
  };
  