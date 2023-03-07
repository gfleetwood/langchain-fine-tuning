from os import environ

from langchain import OpenAI, VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

llm = OpenAI(temperature = 0.0)
embeddings = OpenAIEmbeddings(openai_api_key = environ['OPENAI_API_KEY'])
persist_directory = 'db'
docsearch = Chroma(persist_directory = persist_directory, embedding_function = embeddings)
qa = VectorDBQA.from_chain_type(llm = llm, chain_type = "stuff", vectorstore = docsearch)

while True:
  query = input("Query: ")
  print(qa.run(query))
