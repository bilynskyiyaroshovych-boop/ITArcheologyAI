document.addEventListener('DOMContentLoaded', () => {
    

    const fileInput = document.getElementById('file-input');
    const previewImg = document.getElementById('preview');
    const uploadText = document.getElementById('upload-text');
    const resultBox = document.getElementById('result-box');
    const container = document.querySelector('.container');


    if (fileInput) {
        fileInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    if(previewImg) {
                        previewImg.src = e.target.result;
                        previewImg.style.display = 'block'; 
                        
                        if(uploadText) {
                            uploadText.style.display = 'none'; 
                        }
                    }
                }
                reader.readAsDataURL(file);
            }
        });
    }


    if (typeof gsap !== 'undefined') {
        gsap.from(container, { 
            duration: 1, 
            y: -30, 
            opacity: 0, 
            ease: "power3.out" 
        });

       
        if (resultBox) {
            gsap.fromTo(resultBox, 
                { y: 20, opacity: 0 }, 
                { y: 0, opacity: 1, duration: 0.8, delay: 0.2, ease: "power2.out" }
            );
        }
    }
});