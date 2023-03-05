'''
Take a set of proprietary documents
Split them up into smaller chunks
Create an embedding for each document

Create an embedding for the query
Find the most similar documents in the embedding space
Pass those documents, along with the original query, into a language model to generate an answer

https://langchain.readthedocs.io/en/latest/modules/indexes/vectorstore_examples/chroma.html?highlight=save#persist-the-database
'''

from io import StringIO
import sys
from typing import Dict, Optional
import os

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents.tools import Tool
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader, UnstructuredPDFLoader, OnlinePDFLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA

llm = OpenAI(temperature = 0.0)

# loader = DirectoryLoader('./data/vader_lit/', glob = '**/*.pdf')
# loader = UnstructuredHTMLLoader("./data/Palantir Technologies (PLTR) Q4 2022 Earnings Call Transcript _ The Motley Fool.html")
#loader = UnstructuredPDFLoader("./data/Calculus_Volume_1_-_WEB_68M1Z5W.pdf")
data = loader.load()

text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
texts = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(openai_api_key = os.environ['OPENAI_API_KEY'])
persist_directory = 'db'

docsearch = Chroma.from_documents(documents = texts, embedding = embeddings, persist_directory = persist_directory)
docsearch.persist()
