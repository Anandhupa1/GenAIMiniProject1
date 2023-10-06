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


#1. extract text from pdf
pdf_text = ""
with open('data.pdf', 'rb') as file:
    
    pdf_reader = PdfReader(file)
    
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    
    print("pdf read successfull")
    
# pdf_text1 = "hello my name is annadhu, I am from kerala  "
#2.  splitting pdf into smaller chunks 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
)
chunks = text_splitter.split_text(text=pdf_text);
# print(chunks)

#3. convert it to embeddings with langchain openai embeeddings

embeddings = OpenAIEmbeddings();

# create qdrant client 
os.environ["QDRANT_API_KEY"]="gGo46wXaDANDnGnCkp1GWBChHgbLAOvfaj_OfNqNXldsBcinacVY_w"
os.environ["QDRANT_HOST"]="https://e85b827e-52be-4c6a-b2c1-b4c18c3046bf.us-east4-0.gcp.cloud.qdrant.io:6333"


client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key= os.getenv("QDRANT_API_KEY")
)


from qdrant_client.models import Distance, VectorParams

client.recreate_collection(
    collection_name="collection1",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

#---------------------------------------------------------------------------------
# embeddings = OpenAIEmbeddings();
# vectorStore =  Qdrant(
#     client=client,
#     collection_name="collection1",
#     embeddings=embeddings
# )
os.environ["OPENAI_API_KEY"]="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"


vector_store = Qdrant(
    client=client,
    collection_name="collection1",
    embeddings=OpenAIEmbeddings(),
)
print(vector_store)



vector_store.add_texts(chunks)   # addings docs to vectorstore
print("successfully updated")

#-------working solution----------------------------------------------
# qa = RetrievalQA.from_chain_type(
#     llm =OpenAI(),
#     chain_type = "stuff",
#     retriever = vector_store.as_retriever(),

# )
user_input = "who was bhudha ?"
# answer = qa.run(user_input)

# print(answer)
#----------------------------------------------------------------

found_docs =vector_store.similarity_search(user_input)
print(found_docs)
for x in found_docs :
    print(x)
    print("_______________________________________________________")


llm = OpenAI(model_name="gpt-3.5-turbo",temperature=1.5)
chain = load_qa_chain(llm=llm,chain_type="stuff");
response = chain.run(input_documents=found_docs,question=user_input)

print(response)














# app = Flask(__name__)

# @app.route('/')
# def hello():
#     return 'Hello, Flask!'

# if __name__ == '__main__':
#     app.run(debug=True)



