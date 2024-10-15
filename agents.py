
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from tools import get_analysis_tool
from prompts import get_supervisor_prompt_template
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from custom_callback_handler import CustomStreamlitCallbackHandler


def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    """Creates an agent using the specified ChatOpenAI model, tools, and system prompt."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)


def supervisor_agent(state):
    """The Supervisor Agent manages the flow of the analysis and ensures the right tools are executed."""
    llm = ChatOpenAI(api_key=state["config"]["OPENAI_API_KEY"], temperature=0.1)
    supervisor_prompt = get_supervisor_prompt_template()

    supervisor = create_agent(llm, tools=[], system_prompt=supervisor_prompt)

    chat_history = state.get("messages", [])
    if not chat_history:
        chat_history.append(HumanMessage(content=state["user_input"]))

    output = supervisor.invoke({"messages": chat_history})
    state["next_step"] = output.next_action
    return state


def analysis_agent(state, analysis_type: str):
    """This agent performs the actual data analysis based on the analysis type chosen."""
    analysis_tool = get_analysis_tool(analysis_type)
    data = state["data"]
    result = analysis_tool(data)
    state["analysis_result"] = result
    return state


def define_graph():
    """Define the workflow graph for how the agents interact with each other."""
    workflow = {
        "Supervisor": supervisor_agent,
        "Descriptive Analysis": lambda state: analysis_agent(state, "Descriptive Analysis"),
        "Diagnostic Analysis": lambda state: analysis_agent(state, "Diagnostic Analysis"),
        "Predictive Analysis": lambda state: analysis_agent(state, "Predictive Analysis"),
        "Prescriptive Analysis": lambda state: analysis_agent(state, "Prescriptive Analysis"),
        "Detailed EDA": lambda state: analysis_agent(state, "Detailed EDA"),
    }

    return workflow
