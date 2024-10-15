const wrapper = document.querySelector('.wrapper');
const loginLink = document.querySelector('.login-link');
const registerLink = document.querySelector('.register-link');
const btnPopup = document.querySelector('.btnLogin-popup');
const iconClose = document.querySelector('.icon-close');

registerLink.addEventListener('click', ()=>{
    wrapper.classList.add('active');

});

loginLink.addEventListener('click', ()=>{
    wrapper.classList.remove('active');

});

btnPopup.addEventListener('click', ()=>{
    wrapper.classList.add('active-popup');

});

iconClose.addEventListener('click', ()=>{
    wrapper.classList.remove('active-popup');

});


// JavaScript for slideshow
let slideIndex = 0;
showSlides();

function showSlides() {
    let slides = document.querySelectorAll('.slide');
    // Hide all slides
    slides.forEach(slide => slide.classList.remove('active'));
    
    // Show the current slide
    slideIndex++;
    if (slideIndex > slides.length) {
        slideIndex = 1;
    }
    slides[slideIndex - 1].classList.add('active');
    
    // Change slide every 3 seconds (3000 milliseconds)
    setTimeout(showSlides, 3000);
}
let scrollAmount = 0;

function scrollLeft() {
    const slider = document.querySelector('.reviews-slider');
    scrollAmount -= slider.clientWidth;
    slider.style.transform = `translateX(${scrollAmount}px)`;
}

function scrollRight() {
    const slider = document.querySelector('.reviews-slider');
    scrollAmount += slider.clientWidth;
    slider.style.transform = `translateX(${scrollAmount}px)`;
}


// JavaScript to handle multiple selections
