import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from os import environ
from langchain.vectorstores import Chroma
from langchain import VectorDBQA

def load_chain():    
  llm = OpenAI(temperature = 0.0)
  embeddings = OpenAIEmbeddings(openai_api_key = environ['OPENAI_API_KEY'])
  persist_directory = 'db'
  docsearch = Chroma(persist_directory = persist_directory, embedding_function = embeddings)
  qa = VectorDBQA.from_chain_type(llm = llm, chain_type = "stuff", vectorstore = docsearch)
  
  return(qa)

chain = load_chain()

title = "Palantir Earnings Call: 2022-Q4"
st.set_page_config(page_title = title, page_icon = ":robot:")
st.header(title)

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

user_input = st.text_input("Enter Question: ", key = "input")

if user_input:
    output = chain({"query": user_input})
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output['result'])

if st.session_state["generated"]:
    #for i in range(0, len(st.session_state["generated"]), 1):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["past"][i], is_user = True, key = str(i) + "_user")
        message(st.session_state["generated"][i], key = str(i))
