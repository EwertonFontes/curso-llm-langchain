
from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

import arxiv
import os

llm = Groq(model="llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY"))

def consulta_artigos(title: str) -> str:
    busca = arxiv.Search(
        query=title,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance
    )

    resultados = [
        f"Titulo: {artigo.title}\n"
        f"Categoria: {artigo.primary_category}\n"
        f"Link: {artigo.entry_id}\n"
        for artigo in busca.results()
    ]

    return "\n\n".join(resultados)

consulta_artigos_tool = FunctionTool.from_defaults(fn=consulta_artigos)

agent_worker =  FunctionCallingAgentWorker.from_tools(
    tools=[consulta_artigos_tool],
    verbose=True,
    allow_parallel_tool_calls=False,
    llm=llm
)

agent = AgentRunner(agent_worker)
response = agent.chat("Me retorne artigos sobre LangChain na educação")