from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.tools.tavily_research import TavilyToolSpec

import os

llm = Groq(model="llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY"))


tavily_tool = TavilyToolSpec(
    api_key=os.environ.get("TAVILY_API_KEY")
)

#tavily_tool_list = tavily_tool.to_tool_list()
#tavily_tool.search("Me retorne artigos cientificos sobre LangChain", max_results=3)

tavily_tool_function = FunctionTool.from_defaults(
    fn=tavily_tool.search,
    name="Tavily Search",
    description="Busca artigos com Tavily, sobre determinado tópico"
)

agent_worker = FunctionCallingAgentWorker.from_tools(
    tools=[tavily_tool_function],
    verbose=True,
    allow_parallel_tool_calls=False,
    llm=llm
)

agent = AgentRunner(agent_worker)
resonse = agent.chat("Me retorne artigos sobre LLM e LangChain")
