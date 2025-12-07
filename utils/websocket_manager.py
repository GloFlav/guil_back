import logging
import asyncio
from fastapi import WebSocket
from typing import List, Dict, Any
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Diffuse un message à tous les clients connectés"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

# Instance globale
connection_manager = ConnectionManager()

# --- NOUVEAU : HANDLER DE LOGS ---
class WebSocketLogHandler(logging.Handler):
    """
    Intercepte les logs Python et les envoie via WebSocket
    """
    def __init__(self, manager: ConnectionManager):
        super().__init__()
        self.manager = manager
        # Format exact demandé : Date - Module - Level - Message
        self.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        try:
            # Ne pas logger les logs du WebSocket lui-même pour éviter une boucle infinie
            if "websocket" in record.name.lower() or "uvicorn" in record.name.lower():
                return

            log_entry = self.format(record)
            
            # Créer le payload
            message = {
                "type": "log",
                "level": record.levelname,
                "message": log_entry, # Le message formaté complet
                "timestamp": record.created
            }

            # Envoyer de manière asynchrone dans la boucle d'événements
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.create_task(self.manager.broadcast(message))
            except RuntimeError:
                pass
                
        except Exception:
            self.handleError(record)

class ProgressStreamer:
    def __init__(self, manager: ConnectionManager, websocket: WebSocket):
        self.manager = manager
        self.websocket = websocket

    async def send_progress(self, message: str, status: str, percentage: int = 0, data: Dict = None):
        payload = {
            "type": "progress",
            "status": status,
            "message": message,
            "percentage": percentage,
            "data": data
        }
        await self.websocket.send_json(payload)

    async def send_error(self, message: str, error_code: str):
        payload = {
            "type": "error",
            "status": "error",
            "message": message,
            "error": error_code
        }
        await self.websocket.send_json(payload)

    async def send_result(self, data: Dict, status: str = "complete"):
        payload = {
            "type": "result",
            "status": status,
            "data": data
        }
        await self.websocket.send_json(payload)