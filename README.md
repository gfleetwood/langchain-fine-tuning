# Introduction

This is a template for fine tuning Chat GPT on new data using langchain. You need to have an OpenAI account and an api key accessible as an environment variable called:

OPENAI_API_KEY

# Data

This folder is where you put the the data for fine tuning. The structure will depend on what you what to train, but it's easy to adjust given langchain's different data loaders. For example, if you want to train on just a single pdf just drop it there.

For the next step there is a fork in the road.

# Pinecone

This is the recommended path for scalability as it is optimal for both local and cloud usage. To use the pinecone vector database for training and inference sign up at:

https://www.pinecone.io/

As above, you need to have the following environment variables:

PINECONE_API_KEY

PINECONE_API_ENV

`train.py` does the finetuning. `query.py` provides a command line interface to query the data while app.py allows you to query the data via a streamlit app.

This app can be easily deployed on a platform like streamlit hub.

# Chroma

Chroma is an open souce vector database which stores the generated embeddings locally in a specified folder. Besides that the structure outlined for Pinecone carries over. Given the locally stored embeddings, the size of folder can balloon to many megabytes, so deploying is more complicated.

