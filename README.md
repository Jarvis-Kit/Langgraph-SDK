### Langgraph Python SDK for AgentKit

This is a Python SDK for AgentKit, a framework for building agents.

### Installation

```bash
uv venv && uv sync
```

### Usage

```python
import os
from agentkit import init_runtime, SocketConfig

if __name__ == "__main__":
  # Initialize runtime and wait for connection
  success = init_runtime(
      agent_name="test_agent",
      runtime_endpoint=os.getenv('AGENT_RUNTIME_ENDPOINT') or 'http://localhost:8000',
      socket_config=SocketConfig(
          url=os.getenv('AGENT_RUNTIME_ENDPOINT') or 'http://localhost:8000',
          auth={ "token": os.getenv('AGENT_RUNTIME_TOKEN') or "" }
      ),
      state_whitelist=[...], # Optional
  )

  if not success:
      print("Failed to initialize agent runtime. Exiting...")
      return
  
  # Now, init your program here
```

### Run example for testing

```bash
uv run test.py
```