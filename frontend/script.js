// Theme Toggle Functionality
document.addEventListener('DOMContentLoaded', function() {
    const themeToggle = document.getElementById('themeToggle');
    const body = document.body;
    const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');

    // Function to set theme
    function setTheme(theme) {
        if (theme === 'dark') {
            body.setAttribute('data-theme', 'dark');
            body.classList.remove('light-mode');
            if (themeToggle) {
                themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
            }
            localStorage.setItem('theme', 'dark');
        } else {
            body.setAttribute('data-theme', 'light');
            body.classList.add('light-mode');
            if (themeToggle) {
                themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
            }
            localStorage.setItem('theme', 'light');
        }
    }

    // Toggle theme function
    function toggleTheme() {
        const currentTheme = localStorage.getItem('theme') || (prefersDarkScheme.matches ? 'dark' : 'light');
        setTheme(currentTheme === 'dark' ? 'light' : 'dark');
    }

    // Set initial theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        setTheme(savedTheme);
    } else if (prefersDarkScheme.matches) {
        setTheme('dark');
    } else {
        setTheme('light');
    }

    // Add event listener to theme toggle button
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }

    // Listen for system theme changes
    prefersDarkScheme.addListener((e) => {
        if (!localStorage.getItem('theme')) {
            setTheme(e.matches ? 'dark' : 'light');
        }
    });
});

// Smooth Scrolling for Anchor Links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const targetId = this.getAttribute('href');
        if (targetId === '#') return;
        
        const targetElement = document.querySelector(targetId);
        if (targetElement) {
            window.scrollTo({
                top: targetElement.offsetTop - 80,
                behavior: 'smooth'
            });
        }
    });
});

// Add active class to current nav item
const currentLocation = location.href;
const navLinks = document.querySelectorAll('.nav-links a');
const navLength = navLinks.length;

for (let i = 0; i < navLength; i++) {
    if (navLinks[i].href === currentLocation) {
        navLinks[i].classList.add('active');
    }
}

// Animation on Scroll
const animateOnScroll = () => {
    const elements = document.querySelectorAll('.stat-card, .chart-container, .energy-sources, .about-card, .tech-item, .timeline-item, .team-member');
    
    elements.forEach(element => {
        const elementPosition = element.getBoundingClientRect().top;
        const screenPosition = window.innerHeight / 1.3;
        
        if (elementPosition < screenPosition) {
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }
    });
};

// Set initial state for animation
document.addEventListener('DOMContentLoaded', () => {
    const elements = document.querySelectorAll('.stat-card, .chart-container, .energy-sources, .about-card, .tech-item, .timeline-item, .team-member');
    elements.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    });
    
    // Trigger animation on load
    setTimeout(animateOnScroll, 300);
    
    // Skip animateStats for dashboard - values are updated dynamically from backend
    // Stats will be updated via API calls in index.html
});

// Animate stats counting up
function animateStats() {
    const statValues = document.querySelectorAll('.stat-value');
    
    statValues.forEach(stat => {
        const target = parseInt(stat.textContent);
        const suffix = stat.querySelector('span') ? stat.querySelector('span').textContent : '';
        let current = 0;
        const increment = target / 50; // Adjust speed here
        
        const updateCount = () => {
            if (current < target) {
                current += increment;
                if (current > target) current = target;
                
                stat.textContent = Math.floor(current).toLocaleString() + ' ';
                if (suffix) {
                    const span = document.createElement('span');
                    span.textContent = suffix;
                    stat.appendChild(span);
                }
                
                requestAnimationFrame(updateCount);
            }
        };
        
        // Start the animation when the element is in viewport
        const observer = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting) {
                updateCount();
                observer.unobserve(stat);
            }
        });
        
        observer.observe(stat);
    });
}

// Listen for scroll events
window.addEventListener('scroll', animateOnScroll);

// Form Validation for Contact Form (if exists)
const contactForm = document.getElementById('contactForm');
if (contactForm) {
    contactForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Simple validation
        const name = document.getElementById('name');
        const email = document.getElementById('email');
        const message = document.getElementById('message');
        let isValid = true;
        
        // Reset error messages
        document.querySelectorAll('.error-message').forEach(el => el.remove());
        
        // Name validation
        if (!name.value.trim()) {
            showError(name, 'Name is required');
            isValid = false;
        }
        
        // Email validation
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!email.value.trim()) {
            showError(email, 'Email is required');
            isValid = false;
        } else if (!emailRegex.test(email.value)) {
            showError(email, 'Please enter a valid email');
            isValid = false;
        }
        
        // Message validation
        if (!message.value.trim()) {
            showError(message, 'Message is required');
            isValid = false;
        }
        
        if (isValid) {
            // Here you would typically send the form data to a server
            alert('Thank you for your message! We will get back to you soon.');
            this.reset();
        }
    });
    
    function showError(input, message) {
        const error = document.createElement('div');
        error.className = 'error-message';
        error.style.color = '#ff4444';
        error.style.fontSize = '0.8rem';
        error.style.marginTop = '0.25rem';
        error.textContent = message;
        
        input.parentNode.appendChild(error);
        input.style.borderColor = '#ff4444';
        
        // Remove error on input
        input.addEventListener('input', function onInput() {
            error.remove();
            input.style.borderColor = '';
            input.removeEventListener('input', onInput);
        });
    }
}

// Add loading animation to buttons with loading class
document.querySelectorAll('.btn-loading').forEach(button => {
    button.addEventListener('click', function() {
        this.classList.add('loading');
        this.disabled = true;
        
        // Simulate loading (replace with actual async operation)
        setTimeout(() => {
            this.classList.remove('loading');
            this.disabled = false;
        }, 2000);
    });
});

// Back to top button
const backToTopButton = document.createElement('button');
backToTopButton.id = 'backToTop';
backToTopButton.innerHTML = '<i class="fas fa-arrow-up"></i>';
document.body.appendChild(backToTopButton);

// Show/hide back to top button based on scroll position
window.addEventListener('scroll', () => {
    if (window.pageYOffset > 300) {
        backToTopButton.style.display = 'flex';
    } else {
        backToTopButton.style.display = 'none';
    }
});

// Scroll to top when button is clicked
backToTopButton.addEventListener('click', () => {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
});

// Initialize tooltips
const tooltipElements = document.querySelectorAll('[data-tooltip]');
tooltipElements.forEach(element => {
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = element.getAttribute('data-tooltip');
    
    element.style.position = 'relative';
    element.appendChild(tooltip);
    
    element.addEventListener('mouseenter', () => {
        tooltip.style.visibility = 'visible';
        tooltip.style.opacity = '1';
    });
    
    element.addEventListener('mouseleave', () => {
        tooltip.style.visibility = 'hidden';
        tooltip.style.opacity = '0';
    });
});

// Add CSS for tooltips
const tooltipStyle = document.createElement('style');
tooltipStyle.textContent = `
    .tooltip {
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-size: 0.8rem;
        white-space: nowrap;
        pointer-events: none;
        opacity: 0;
        visibility: hidden;
        transition: all 0.2s ease;
        z-index: 1000;
        margin-bottom: 0.5rem;
    }
    
    .tooltip::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        border-width: 5px;
        border-style: solid;
        border-color: rgba(0, 0, 0, 0.8) transparent transparent transparent;
    }
`;
document.head.appendChild(tooltipStyle);

// Add CSS for back to top button
const backToTopStyle = document.createElement('style');
backToTopStyle.textContent = `
    #backToTop {
        display: none;
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: var(--primary);
        color: white;
        border: none;
        cursor: pointer;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        z-index: 999;
        transition: all 0.3s ease;
    }
    
    #backToTop:hover {
        background: var(--primary-dark);
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
`;
document.head.appendChild(backToTopStyle);

// Add CSS for loading animation
const loadingStyle = document.createElement('style');
loadingStyle.textContent = `
    .btn.loading {
        position: relative;
        pointer-events: none;
        color: transparent !important;
    }
    
    .btn.loading::after {
        content: '';
        position: absolute;
        width: 20px;
        height: 20px;
        top: 50%;
        left: 50%;
        margin: -10px 0 0 -10px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-top-color: white;
        border-radius: 50%;
        animation: button-loading 0.6s linear infinite;
    }
    
    @keyframes button-loading {
        to { transform: rotate(360deg); }
    }
`;
document.head.appendChild(loadingStyle);


