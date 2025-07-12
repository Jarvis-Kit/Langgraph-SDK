import ssl
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

@dataclass
class MessageEvent:
    namespace: str
    agent_name: str
    sent_at: str
    message: dict[str, Any]
    config: dict[str, Any]
    
    def __init__(
        self,
        rawEvent: dict[str, Any]
    ):
        # Mapping from typescript object to python object
        message = rawEvent["message"]
        self.namespace = rawEvent["namespace"]
        self.agent_name = rawEvent["agentName"]
        self.sent_at = rawEvent["sentAt"]
        self.message = {
            "id": message["id"],
            "thread": message["thread"],
            "content": message["content"],
            "role": message["role"]
        }
        self.config = rawEvent["config"]

@dataclass
class SocketConfig:
    url: str
    reconnection: bool = True
    reconnection_attempts: int = 10
    reconnection_delay: int = 1
    reconnection_delay_max: int = 5
    
@dataclass
class RabbitMQConfig:
    url: str
    ssl_context: ssl.SSLContext | None = None

@dataclass
class RuntimeMessage(TypedDict):
    id: str
    thread: str
    content: str
    role: Literal["user", "agent"]
    toolCallId: str
    toolName: str
    toolInput: dict[str, Any]
    toolResults: dict[str, Any]
    metadata: dict[str, Any]
    createdAt: str
    updatedAt: str

@dataclass
class ClientResponseData(TypedDict):
    namespace: str
    agent_name: str
    tool_call_id: str
    response: Any