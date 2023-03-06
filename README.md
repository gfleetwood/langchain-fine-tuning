Template for fine tuning Chat GPT on new data. Generally raw_run.py trains while cooked_run.py accesses a commmand line for querying the document. app.py provides a streamlit app for querying the document. 

The chroma directory is the setup for doing this training locally via the open source chroma vector database. The pinecone directory does the same but leverages the cloud pinecone vector database. This makes it conducive for deploying the app to the cloud.

For quick deployment you can use streamlit cloud. 


