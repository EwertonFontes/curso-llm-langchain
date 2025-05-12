from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

import os


llm = Groq(model="llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY"))

def calcular_imposto_renda(rendimento: float) -> str:
    if rendimento <= 2000:
        return "ISENTO"
    elif 2000 < rendimento <= 5000:
        imposto = (rendimento - 2000) * 0.1
    elif 5000 < rendimento <= 10000:
        imposto = (rendimento - 5000) * 0.15 + 300
    else:
        imposto = (rendimento - 10000) * 0.2 + 1050
        
    return f"Imposto devido {imposto:.2f}, base no redimento {rendimento:.2f}"


ferramenta_imposto_renda = FunctionTool.from_defaults(
    fn=calcular_imposto_renda,
    name="Calcular imposto de renda",
    description=(
        "Calcula o imposto de renda com base no rendimento anual."
        "Argumento: rendimento (float)."
        "Retorna o valor do imposto devido de acordo com faixas de rendimento"
    )
)

agent_worker_imposto = FunctionCallingAgentWorker.from_tools(
    tools=[ferramenta_imposto_renda],
    verbose=True,
    allow_parallel_tool_calls=True,
    llm=llm
)

agent_imposto = AgentRunner(agent_worker_imposto)
response = agent_imposto.chat("""
    Qual é o imposto de renda devido por uma pessoa com rendimento anual de R$7500?
"""
)