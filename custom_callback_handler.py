from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema import AgentAction

class CustomStreamlitCallbackHandler(StreamlitCallbackHandler):
    """Custom callback handler for managing interactions between the agent and Streamlit."""

    def __init__(self, parent_container):
        super().__init__()
        self._parent_container = parent_container

    def write_agent_name(self, name: str):
        """Display the name of the current agent performing an action."""
        self._parent_container.write(f"**Current Agent:** {name}")

    def on_agent_action(self, action: AgentAction):
        """Display the agent's actions in Streamlit."""
        self._parent_container.write(f"**Agent is performing:** {action.log}")
        super().on_agent_action(action)

    def on_agent_finish(self, output):
        """Display the agent's final output once the task is complete."""
        self._parent_container.write("### Agent Task Completed")
        self._parent_container.write(output)
