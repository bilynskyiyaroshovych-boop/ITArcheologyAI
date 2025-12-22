document.addEventListener('DOMContentLoaded', () => {
    
    // --- Елементи DOM ---
    const fileInput = document.getElementById('file-input');
    const previewImg = document.getElementById('preview');
    const uploadText = document.getElementById('upload-text');
    const resultBox = document.getElementById('result-box');
    const container = document.querySelector('.container');

    // --- Логіка попереднього перегляду (Preview) ---
    if (fileInput) {
        fileInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    if(previewImg) {
                        previewImg.src = e.target.result;
                        previewImg.style.display = 'block'; // Явно показуємо
                        
                        if(uploadText) {
                            uploadText.style.display = 'none'; // Ховаємо текст
                        }
                    }
                }
                reader.readAsDataURL(file);
            }
        });
    }

    // --- GSAP Анімація ---
    
    // 1. Анімація появи контейнера при завантаженні
    if (typeof gsap !== 'undefined') {
        gsap.from(container, { 
            duration: 1, 
            y: -30, 
            opacity: 0, 
            ease: "power3.out" 
        });

        // 2. Анімація результату (якщо він є на сторінці)
        if (resultBox) {
            gsap.fromTo(resultBox, 
                { y: 20, opacity: 0 }, 
                { y: 0, opacity: 1, duration: 0.8, delay: 0.2, ease: "power2.out" }
            );
        }
    }
});