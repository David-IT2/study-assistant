"""
Builds the tool-calling agent: an LLM that can decide whether to use the
calculator, web search, or document search tools to answer a question.
"""
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import get_all_tools
from llm_config import get_llm


def build_system_prompt(tools) -> str:
    tool_names = {t.name for t in tools}
    lines = ["You are a helpful study and research assistant."]

    if "search_my_documents" in tool_names:
        lines.append(
            "Use the 'search_my_documents' tool when the question could relate to "
            "the user's uploaded documents."
        )
    else:
        lines.append(
            "No documents have been uploaded yet, so you have no document search "
            "tool available. If the user asks about 'their documents,' tell them "
            "no documents have been added yet and suggest they upload some."
        )

    lines.append("Use 'calculator' for any arithmetic.")
    lines.append("Use 'web_search' for current events or anything not covered by other tools.")
    lines.append("Only call tools that are actually available to you. If you use a tool, "
                  "base your answer on what it returns. If no tool is relevant, answer directly.")
    return " ".join(lines)


def build_agent_executor(verbose: bool = False) -> AgentExecutor:
    tools = get_all_tools()
    llm = get_llm(temperature=0)
    system_prompt = build_system_prompt(tools)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=verbose)
