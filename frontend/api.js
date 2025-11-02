// API configuration
function inferDefaultBackend() {
    try {
        return `${window.location.protocol}//${window.location.hostname}:8000`;
    } catch (_) {
        return 'https://voltnet.onrender.com';
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
 * Make a prediction using the backend API
 * @param {Object} data - Prediction request data (renewable_pct, battery_soc, load_factor, baseline_idx)
 * @returns {Promise<Object>} - Backend prediction results with voltage, flows, curtailment_pct, battery_schedule
 */
export async function predictOPF(data) {
    const endpoint = '/predict_opf';
    logApiCall(endpoint, data);
    
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

    // Return real backend data directly - no transformation, no fallback
    return await response.json();
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
