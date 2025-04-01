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

function loadAllContextData() {
    // Load latest query
    fetch('/api/latest-firebase-query')
        .then(response => response.json())
        .then(data => updateLatestQuery(data))
        .catch(error => console.error('Error loading latest query:', error));
    
    // Load health context
    fetch('/api/latest-health-context')
        .then(response => response.json())
        .then(data => updateHealthContext(data))
        .catch(error => console.error('Error loading health context:', error));
    
    // Load work context
    fetch('/api/latest-work-context')
        .then(response => response.json())
        .then(data => updateWorkContext(data))
        .catch(error => console.error('Error loading work context:', error));
    
    // Load commute context
    fetch('/api/latest-commute-context')
        .then(response => response.json())
        .then(data => updateCommuteContext(data))
        .catch(error => console.error('Error loading commute context:', error));
}

function updateLatestQuery(data) {
    const section = document.getElementById('latestQuerySection');
    if (section) {
        if (data.success && data.data) {
            const query = data.data;
            section.innerHTML = `
                <div class="context-item">
                    <p><strong>Query:</strong> ${query.query}</p>
                    <p><small>Time: ${formatTimestamp(query.timestamp)}</small></p>
                </div>
            `;
        } else {
            section.innerHTML = '<p>No queries available</p>';
        }
    }
}

function updateHealthContext(data) {
    const section = document.getElementById('healthContextSection');
    if (section) {
        if (data.success && data.data) {
            const health = data.data;
            section.innerHTML = `
                <div class="context-item">
                    <p><strong>Exercise:</strong> ${health.exerciseMinutes} minutes</p>
                    <p><strong>Blood Sugar:</strong> ${health.bloodSugar}</p>
                    <p><strong>Meal Type:</strong> ${health.mealType}</p>
                    <p><small>Updated: ${formatTimestamp(health.timestamp)}</small></p>
                </div>
            `;
        } else {
            section.innerHTML = '<p>No health context available</p>';
        }
    }
}

function updateWorkContext(data) {
    const section = document.getElementById('workContextSection');
    if (section) {
        if (data.success && data.data) {
            const work = data.data;
            section.innerHTML = `
                <div class="context-item">
                    <p><strong>Task:</strong> ${work.taskName}</p>
                    <p><strong>Status:</strong> ${work.status}</p>
                    <p><strong>Priority:</strong> ${work.priority}</p>
                    <p><small>Deadline: ${formatTimestamp(work.deadline)}</small></p>
                </div>
            `;
        } else {
            section.innerHTML = '<p>No work context available</p>';
        }
    }
}

function updateCommuteContext(data) {
    const section = document.getElementById('commuteContextSection');
    if (section) {
        if (data.success && data.data) {
            const commute = data.data;
            section.innerHTML = `
                <div class="context-item">
                    <p><strong>From:</strong> ${commute.startLocation}</p>
                    <p><strong>To:</strong> ${commute.endLocation}</p>
                    <p><strong>Mode:</strong> ${commute.transportMode}</p>
                    <p><strong>Traffic:</strong> ${commute.trafficCondition}</p>
                    <p><small>Updated: ${formatTimestamp(commute.timestamp)}</small></p>
                </div>
            `;
        } else {
            section.innerHTML = '<p>No commute context available</p>';
        }
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