import os
import openai
import qdrant_client
from flask_cors import CORS
from flask import Flask,request,jsonify;
from langchain.vectorstores import Qdrant;
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings;
from dotenv import load_dotenv, find_dotenv;
from langchain.llms import OpenAI

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ["OPENAI_API_KEY"]

client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key= os.getenv("QDRANT_API_KEY")
)
vector_store = Qdrant(
    client=client,
    collection_name="collection1",
    embeddings=OpenAIEmbeddings(),
)






app = Flask(__name__)
CORS(app)
@app.route('/')
def home():
    return 'Home page!'
@app.route('/query',methods=["post"])
def query():
    data = request.get_json();
    user_input =  data["query"];
    if user_input :
        output ={}
        
        found_docs =vector_store.similarity_search(user_input)
        documents=[]
        for i,x in enumerate(found_docs):
            documents.append(str(x))
        output["documents"]=documents;
        #------------------------------------------
        llm = OpenAI(model_name="gpt-3.5-turbo",temperature=1.5)
        chain = load_qa_chain(llm=llm,chain_type="stuff");
        response = chain.run(input_documents=found_docs,question=user_input)
        print(response)
        output["response"]=str(response);
        return output;
    else: return {error:"no input found .."}





























if __name__ == '__main__':
    app.run(debug=False)



