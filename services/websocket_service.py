import logging
import json
from typing import Set, Callable, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Gère les connexions WebSocket"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        """Accepte une nouvelle connexion WebSocket"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Active connections: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        """Ferme une connexion WebSocket"""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Active connections: {len(self.active_connections)}")
    
    async def broadcast(self, data: dict):
        """Envoie un message à tous les clients connectés"""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"Error broadcasting: {str(e)}")
                disconnected.add(connection)
        
        for connection in disconnected:
            await self.disconnect(connection)
    
    async def send_personal(self, websocket: WebSocket, data: dict):
        """Envoie un message à une connexion spécifique"""
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Error sending personal message: {str(e)}")
            await self.disconnect(websocket)


class ProgressStreamer:
    """Gère le streaming de la progression et des messages"""
    
    def __init__(self, connection_manager: ConnectionManager, websocket: WebSocket):
        self.connection_manager = connection_manager
        self.websocket = websocket
        self.message_history = []
    
    async def send_progress(
        self,
        message: str,
        status: str = "processing",
        data: Optional[dict] = None,
        percentage: int = 0
    ):
        """Envoie une mise à jour de progression"""
        progress_message = {
            "type": "progress",
            "message": message,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "percentage": percentage,
            "data": data or {}
        }
        
        self.message_history.append(progress_message)
        
        logger.info(f"Progress: {message} ({status})")
        
        await self.connection_manager.send_personal(self.websocket, progress_message)
    
    async def send_error(self, error_message: str, error_code: str = "ERROR"):
        """Envoie un message d'erreur"""
        error_msg = {
            "type": "error",
            "message": error_message,
            "status": "error",
            "error_code": error_code,
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_history.append(error_msg)
        
        logger.error(f"Error sent: {error_message}")
        
        await self.connection_manager.send_personal(self.websocket, error_msg)
    
    async def send_result(self, result_data: dict, status: str = "complete"):
        """Envoie le résultat final"""
        result_message = {
            "type": "result",
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "data": result_data
        }
        
        self.message_history.append(result_message)
        
        logger.info(f"Result sent: {status}")
        
        await self.connection_manager.send_personal(self.websocket, result_message)
    
    def get_history(self) -> list:
        """Retourne l'historique des messages"""
        return self.message_history


# Instance globale du gestionnaire de connexions
connection_manager = ConnectionManager()