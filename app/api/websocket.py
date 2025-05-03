from flask import Blueprint
from flask_sock import Sock
from ..utils.websocket import ws_manager

# Create blueprint
ws = Blueprint('websocket', __name__)

def init_websocket(app):
    """Initialize WebSocket routes."""
    sock = Sock(app)
    
    @sock.route('/ws')
    def handle_websocket(ws):
        """Handle WebSocket connections."""
        ws_manager.add_connection(ws)
        try:
            while True:
                # Keep the connection alive
                ws.receive()
        except Exception:
            ws_manager.remove_connection(ws) 