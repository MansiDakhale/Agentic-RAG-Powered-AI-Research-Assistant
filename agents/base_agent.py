# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import json
from datetime import datetime

class AgentMessage:
    """Standard message format between agents"""
    def __init__(self, sender: str, receiver: str, content: Any, message_type: str):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.message_type = message_type
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "message_type": self.message_type,
            "timestamp": self.timestamp.isoformat()
        }

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, llm, vector_store=None):
        self.name = name
        self.llm = llm
        self.vector_store = vector_store
        self.message_history: List[AgentMessage] = []
    
    @abstractmethod
    def execute(self, message: AgentMessage) -> AgentMessage:
        """Execute the agent's main functionality"""
        pass
    
    def send_message(self, receiver: str, content: Any, message_type: str) -> AgentMessage:
        """Send message to another agent"""
        message = AgentMessage(self.name, receiver, content, message_type)
        self.message_history.append(message)
        return message
    
    def log_activity(self, activity: str):
        """Log agent activity for debugging"""
        print(f"[{self.name}] {activity}")