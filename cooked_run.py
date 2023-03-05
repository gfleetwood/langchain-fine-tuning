from io import StringIO
import sys
from typing import Dict, Optional
from os import environ

from langchain import OpenAI, VectorDBQA
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents.tools import Tool
#from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

llm = OpenAI(temperature = 0.0)
embeddings = OpenAIEmbeddings(openai_api_key = environ['OPENAI_API_KEY'])
persist_directory = 'db'
docsearch = Chroma(persist_directory = persist_directory, embedding_function = embeddings)
qa = VectorDBQA.from_chain_type(llm = llm, chain_type = "stuff", vectorstore = docsearch)

query = "How do I differentiate log(x)?"
print(qa.run(query))
