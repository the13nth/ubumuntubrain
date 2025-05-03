import threading
import json

class WebSocketManager:
    """Manage WebSocket connections and broadcasting."""
    
    def __init__(self):
        self.connections = set()
        self.lock = threading.Lock()
    
    def add_connection(self, ws):
        """Add a new WebSocket connection."""
        with self.lock:
            self.connections.add(ws)
    
    def remove_connection(self, ws):
        """Remove a WebSocket connection."""
        with self.lock:
            self.connections.discard(ws)
    
    def broadcast(self, message):
        """Broadcast a message to all connected WebSocket clients."""
        with self.lock:
            dead_sockets = set()
            for ws in self.connections:
                try:
                    ws.send(json.dumps(message))
                except Exception:
                    dead_sockets.add(ws)
            
            # Remove dead connections
            for dead_ws in dead_sockets:
                self.connections.remove(dead_ws)

# Create a global instance
ws_manager = WebSocketManager() 