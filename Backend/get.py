import os 
import openai
import qdrant_client
from qdrant_client.http import models
from langchain.vectorstores import Qdrant;
from langchain.chains.question_answering import load_qa_chain

from PyPDF2 import PdfReader
from flask import Flask
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv;
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ["OPENAI_API_KEY"]
#--------------------------------------------------------------------------------------


client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key= os.getenv("QDRANT_API_KEY")
)



vector_store = Qdrant(
    client=client,
    collection_name="collection1",
    embeddings=OpenAIEmbeddings(),
)
print(vector_store)


user_input = "who was bhudha ?"

found_docs =vector_store.similarity_search(user_input)
print(found_docs)
for x in found_docs :
    print(x)
    print("_______________________________________________________")


llm = OpenAI(model_name="gpt-3.5-turbo",temperature=1.5)
chain = load_qa_chain(llm=llm,chain_type="stuff");
response = chain.run(input_documents=found_docs,question=user_input)

print(response)