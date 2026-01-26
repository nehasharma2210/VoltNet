// API configuration with backend wake-up
function inferDefaultBackend() {
    return 'https://voltnet.onrender.com';  // ✅ Your backend URL
}

// Wake up backend on page load
window.addEventListener('load', () => {
    console.log("🚀 Waking up backend...");
    wakeUpBackend();
});

// Function to wake up sleeping backend
async function wakeUpBackend() {
    const backendUrl = getConfiguredBackendUrl();
    try {
        console.log("📡 Pinging backend:", backendUrl);
        const response = await fetch(`${backendUrl}/health`, {
            method: 'GET',
            headers: { 'Accept': 'application/json' }
        });
        
        if (response.ok) {
            console.log("✅ Backend is awake!");
            return true;
        } else {
            console.log("⏳ Backend waking up... trying again in 5s");
            setTimeout(wakeUpBackend, 5000);
        }
    } catch (error) {
        console.log("⏳ Backend sleeping... trying again in 5s");
        setTimeout(wakeUpBackend, 5000);
    }
}


function getConfiguredBackendUrl() {
    try {
        const fromWindow = typeof window !== 'undefined' && window.BACKEND_URL ? window.BACKEND_URL : null;
        const fromQuery = typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('backend') : null;
        const fromStorage = typeof window !== 'undefined' ? window.localStorage.getItem('BACKEND_URL') : null;
        return fromWindow || fromQuery || fromStorage || inferDefaultBackend();
    } catch (_) {
        return inferDefaultBackend();
    }
}

const API_BASE_URL = getConfiguredBackendUrl();

// Debug function to log API calls
function logApiCall(endpoint, data) {
    console.log(`API Call: ${endpoint}`, {
        timestamp: new Date().toISOString(),
        data: data
    });
}

// Debug function to log API responses
function logApiResponse(endpoint, response, isError = false) {
    const logMethod = isError ? console.error : console.log;
    logMethod(`API Response [${endpoint}]:`, {
        timestamp: new Date().toISOString(),
        status: response?.status,
        statusText: response?.statusText,
        response: response
    });
}

/**
 * Make a prediction using the backend API with retry logic
 * @param {Object} data - Prediction request data
 * @returns {Promise<Object>} - Backend prediction results
 */
export async function predictOPF(data) {
    const endpoint = '/predict_opf';
    logApiCall(endpoint, data);
    
    // First try to wake up backend
    await wakeUpBackend();
    
    // Retry logic for prediction
    for (let attempt = 1; attempt <= 3; attempt++) {
        try {
            const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(data)
            });

            logApiResponse(endpoint, response);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            return await response.json();
            
        } catch (error) {
            console.log(`Attempt ${attempt} failed:`, error.message);
            if (attempt < 3) {
                console.log("⏳ Retrying in 3 seconds...");
                await new Promise(resolve => setTimeout(resolve, 3000));
            } else {
                throw error;
            }
        }
    }
}

/**
 * Check if the backend is healthy
 * @returns {Promise<boolean>} - True if backend is healthy
 */
export async function checkBackendHealth() {
    const endpoint = '/health';
    logApiCall(endpoint);
    
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });

        logApiResponse(endpoint, response);

        if (!response.ok) {
            throw new Error(`Health check failed with status: ${response.status}`);
        }

        const data = await response.json();
        const isHealthy = data.status === 'healthy' && data.loaded === true;
        
        if (!isHealthy) {
            console.warn('Backend health check returned unhealthy status:', data);
        }
        
        return isHealthy;
    } catch (error) {
        console.error('Health check failed:', error);
        return false;
    }
}
