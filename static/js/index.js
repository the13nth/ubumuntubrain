// Initialize Firebase
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-app.js";
import { getDatabase, ref, onValue, get, child } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-database.js";

// Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyAmHP7pmyySEOoO9ZlmoCAdKW3iDVGjC1c",
    authDomain: "ubumuntu-8d53c.firebaseapp.com",
    projectId: "ubumuntu-8d53c",
    storageBucket: "ubumuntu-8d53c.firebasestorage.app",
    messagingSenderId: "894659655360",
    appId: "1:894659655360:web:d289cbf449f789f89e9f25",
    measurementId: "G-MJGP10SWBX",
    databaseURL: "https://ubumuntu-8d53c-default-rtdb.firebaseio.com"
};

// Initialize Firebase app
const app = initializeApp(firebaseConfig);
const database = getDatabase(app);

// WebSocket connection for real-time chat
let ws;

document.addEventListener('DOMContentLoaded', function() {
    // Setup navigation
    setupNavigation();
    
    // Initialize WebSocket connection
    setupWebSocket();
    
    // Setup chat functionality
    setupChat();
    
    // Load all context data
    loadAllContextData();
    
    // Set up periodic data refresh
    setInterval(loadAllContextData, 30000); // Refresh every 30 seconds
});

function setupWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onopen = () => {
        console.log('WebSocket connection established');
        // Send periodic ping to keep connection alive
        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'pong') {
            console.log('Received pong from server');
        } else {
            handleWebSocketMessage(data);
        }
    };
    
    ws.onclose = () => {
        console.log('WebSocket connection closed. Attempting to reconnect...');
        setTimeout(setupWebSocket, 5000);
    };
}

function setupChat() {
    const chatInput = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendMessage');
    const chatMessages = document.getElementById('chatMessages');
    
    if (chatInput && sendButton && chatMessages) {
        sendButton.addEventListener('click', () => sendMessage());
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    }
}

function sendMessage() {
    const chatInput = document.getElementById('chatInput');
    const chatMessages = document.getElementById('chatMessages');
    
    if (chatInput && chatInput.value.trim()) {
        const message = chatInput.value.trim();
        
        // Add user message to chat
        appendMessage('user', message);
        
        // Clear input
        chatInput.value = '';
        
        // Send message to server
        fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: message })
        })
        .then(response => response.json())
        .then(data => {
            appendMessage('assistant', data.answer);
            // Reload context data after receiving response
            loadAllContextData();
        })
        .catch(error => {
            console.error('Error:', error);
            appendMessage('system', 'Error processing your request. Please try again.');
        });
    }
}

function appendMessage(role, content) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}-message`;
    messageDiv.innerHTML = `<p>${content}</p>`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function handleWebSocketMessage(data) {
    if (data.type === 'api_call') {
        console.log('API call made:', data.query);
        loadAllContextData(); // Refresh data when API calls are made
    }
}

// Utility Functions
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatTimestamp(timestamp) {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    return date.toLocaleString();
}

function showError(message) {
    const errorContainer = document.getElementById('error-container') || createErrorContainer();
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `
        ${escapeHtml(message)}
        <button onclick="this.parentElement.remove()">×</button>
    `;
    errorContainer.appendChild(errorDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentElement) {
            errorDiv.remove();
        }
    }, 5000);
}

function createErrorContainer() {
    const container = document.createElement('div');
    container.id = 'error-container';
    document.body.appendChild(container);
    return container;
}

function showLoading(show = true) {
    const loader = document.getElementById('loading-indicator');
    if (loader) {
        loader.style.display = show ? 'block' : 'none';
    }
}

// Data Loading Functions
async function loadAllContextData() {
    showLoading(true);
    try {
        const [queryResponse, healthResponse, workResponse, commuteResponse] = await Promise.all([
            fetch('/api/latest-firebase-query'),
            fetch('/api/latest-health-context'),
            fetch('/api/latest-work-context'),
            fetch('/api/latest-commute-context')
        ]);

        const responses = {
            query: await queryResponse.json(),
            health: await healthResponse.json(),
            work: await workResponse.json(),
            commute: await commuteResponse.json()
        };

        Object.entries(responses).forEach(([type, data]) => {
            if (!data.success) {
                throw new Error(`Failed to load ${type} data: ${data.error || 'Unknown error'}`);
            }
        });

        updateLatestQuery(responses.query.data);
        updateHealthContext(responses.health.data);
        updateWorkContext(responses.work.data);
        updateCommuteContext(responses.commute.data);

    } catch (error) {
        console.error('Error loading context data:', error);
        showError(error.message || 'Failed to load context data');
    } finally {
        showLoading(false);
    }
}

// UI Update Functions
function updateLatestQuery(data) {
    const container = document.getElementById('latest-query');
    if (!container) return;

    const query = data?.query || 'No query available';
    const timestamp = data?.timestamp;
    const status = data?.processed ? 'Processed' : 'Pending';
    const statusClass = data?.processed ? 'success' : 'pending';

    container.innerHTML = `
        <div class="query-card">
            <h3>Latest Query</h3>
            <div class="query-text">${escapeHtml(query)}</div>
            <div class="query-timestamp">Timestamp: ${formatTimestamp(timestamp)}</div>
            <div class="query-status ${statusClass}">${status}</div>
            ${!data?.processed ? '<button onclick="processQuery()">Process Query</button>' : ''}
        </div>
    `;
}

function updateHealthContext(data) {
    const container = document.getElementById('health-context');
    if (!container) return;

    container.innerHTML = `
        <div class="context-card health">
            <h3>Health Context</h3>
            <p>${escapeHtml(data?.context || 'No health context available')}</p>
            <div class="timestamp">Last updated: ${formatTimestamp(data?.timestamp)}</div>
        </div>
    `;
}

function updateWorkContext(data) {
    const container = document.getElementById('work-context');
    if (!container) return;

    container.innerHTML = `
        <div class="context-card work">
            <h3>Work Context</h3>
            <p>${escapeHtml(data?.context || 'No work context available')}</p>
            <div class="timestamp">Last updated: ${formatTimestamp(data?.timestamp)}</div>
        </div>
    `;
}

function updateCommuteContext(data) {
    const container = document.getElementById('commute-context');
    if (!container) return;

    container.innerHTML = `
        <div class="context-card commute">
            <h3>Commute Context</h3>
            <p>${escapeHtml(data?.context || 'No commute context available')}</p>
            <div class="timestamp">Last updated: ${formatTimestamp(data?.timestamp)}</div>
        </div>
    `;
}

async function processQuery() {
    showLoading(true);
    try {
        const response = await fetch('/api/process-query', {
            method: 'POST'
        });
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Failed to process query');
        }
        
        // Reload data after processing
        await loadAllContextData();
        
    } catch (error) {
        console.error('Error processing query:', error);
        showError(error.message || 'Failed to process query');
    } finally {
        showLoading(false);
    }
}

function setupNavigation() {
    const navToggle = document.getElementById('navToggle');
    const navMenu = document.getElementById('navMenu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            navToggle.classList.toggle('active');
        });
    }
}

function formatTimestamp(timestamp) {
    if (!timestamp) return 'Unknown time';
    
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) {
        return 'Just now';
    } else if (diffMins < 60) {
        return `${diffMins} minute${diffMins === 1 ? '' : 's'} ago`;
    } else if (diffMins < 1440) {
        const hours = Math.floor(diffMins / 60);
        return `${hours} hour${hours === 1 ? '' : 's'} ago`;
    } else {
        const options = { 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric', 
            hour: '2-digit', 
            minute: '2-digit' 
        };
        return date.toLocaleDateString('en-US', options);
    }
} 