from typing import Any, cast
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig

from .callback_handler import JarvisKitCallbackHandler
from .classes import SocketConfig, RabbitMQConfig
from .jarvis_runtime import JarvisKitRuntime
from .classes import MessageEvent

# Global runtime instance
_runtime: JarvisKitRuntime | None = None

def init_runtime(
    namespace: str,
    namespace_api_key: str,
    runtime_endpoint: str,
    socket_config: SocketConfig,
    agents: dict[str, CompiledStateGraph[Any, Any, Any]] = {},
    timeout: int = 10,
    max_concurrent_workers: int = 2,
    rabbitmq_config: RabbitMQConfig | None = None

) -> JarvisKitRuntime:
    """Initialize the agent runtime and wait for connection"""
    global _runtime
    
    _runtime = JarvisKitRuntime(
        namespace=namespace,
        namespace_api_key=namespace_api_key,
        runtime_endpoint=runtime_endpoint,
        socket_config=socket_config,
        agents=agents,
        max_concurrent_workers=max_concurrent_workers,
        rabbitmq_config=rabbitmq_config
    )
    
    # Wait for connection to be established
    if _runtime.wait_for_connection(timeout):
        print("Agent runtime initialized successfully")
        print(f"Space name: {_runtime.namespace}")
        print(f"Agents: {', '.join(_runtime.agents.keys())}")
        print("-"*100)
        return _runtime
    else:
        print(f"Failed to initialize agent runtime '{namespace}' within {timeout} seconds")
        exit(1)

def get_runtime() -> JarvisKitRuntime:
    if _runtime is None:
        raise RuntimeError("Agent runtime not initialized")
    return _runtime

async def default_message_handler(agent: CompiledStateGraph[Any, Any, Any], event: MessageEvent) -> bool:
    try:
        config: RunnableConfig = RunnableConfig(
            configurable={
                "thread_id": event.message["thread"],
                "checkpoint_ns": agent.name,
                **event.config
            },
            callbacks=[JarvisKitCallbackHandler(get_runtime(), event.message["thread"])]
        )
        
        await agent.ainvoke(cast(Any, {}), config=config)
        
        return True
    except Exception as e:
        print(f"Task execution failed: {e}")
        return False
