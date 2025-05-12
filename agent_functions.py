from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner, ReActAgent
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex,  StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import os

llm = Groq(model="llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY"))


url = "files/LLM.pdf"
artigo = SimpleDirectoryReader(input_files=[url]).load_data()

url_2 = "files/LLM_2.pdf"
tutorial = SimpleDirectoryReader(input_files=[url_2]).load_data()

#GERAR OS EMBEDDINGS
Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large"
)

artigo_index = VectorStoreIndex.from_documents(artigo)
tutorial_index = VectorStoreIndex.from_documents(tutorial)

artigo_index.storage_context.persist(persist_dir="artigo")
tutorial_index.storage_context.persist(persist_dir="tutorial")

#ENGINE DE BUSCA
storage_context = StorageContext.from_defaults(
    persist_dir="artigo"
)
artigo_index = load_index_from_storage(storage_context)

storage_context = StorageContext.from_defaults(
    persist_dir="tutorial"
)

tutorial_index = load_index_from_storage(storage_context)

artigo_engine = artigo_index.as_query_engine(llm=llm)
tutorial_engine = tutorial_index.as_query_engine(llm=llm)

query_engine_tools = [
    QueryEngineTool(
        query_engine=artigo_engine,
        metadata=ToolMetadata(
            name="artigo_engine",
            description="Fornece informações sobre LLM e LangChain."
            "Use uma pergunta detalhada em texto simples como entrada para a ferramenta"
        )
    ),
     QueryEngineTool(
        query_engine=tutorial_engine,
        metadata=ToolMetadata(
            name="tutoria_engine",
            description="Fornece informações sobre casos de uso e aplicações em LLMs"
            "Use uma pergunta detalhada em texto simples como entrada para a ferramenta"
        )
    ),
]

agent_worker = FunctionCallingAgentWorker.from_tools(
    query_engine_tools,
    verbose=True,
    allow_parallel_tool_calls=True,
    llm=llm
)

agent_document = AgentRunner(agent_worker)
response = agent_document.chat(
    "Quais as principais aplicações posso construir com LLM e Langchain"
)

agent = ReActAgent.from_tools(
    query_engine_tools,
    verbose=True,
    llm=llm
)

response = agent.chat(
    "Quais as principais ferramentas usadas em LangChain?"
)

