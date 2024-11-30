from flask import Flask, render_template, request, jsonify
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from flask import session 
from flask_session import Session
import os
#from langchain_google_genai import GoogleGenerativeAIEmbeddings

# *** X ***** X *** X ***** #

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY","default key")  # Required for Flask sessions
app.config["SESSION_TYPE"] = "filesystem"  # Store sessions in the file system
Session(app)

from dotenv import load_dotenv
load_dotenv()


DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/' 

# *** X ***** X *** X ***** #

def create_vectordb():
    #loader = PyPDFLoader('data\Employee Handbook.pdf')
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents = loader.load()
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print("Process Completed !!.........")

# *** X ***** X *** X ***** #

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=None)

prompt_template = """You are a helpful, friendly and honest chatbot named "WorkMate" who always greet the user at the begining and is talkative. 
When the user asks the question use the following information to answer factually correct answers in presentable manner.
Don't mention company name.
If you don't know the answer, clearly say you don't know the answer.

Context: {context}
Question: {question}

Only return the correct answer below and nothing else.
Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def generate_query(query):
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever, chain_type="stuff", chain_type_kwargs={"prompt": PROMPT})
    response = qa({"query": query})
    return response['result']


# *** X ***** X *** X *****

@app.route("/")
def home():
    return render_template('landing.html')

@app.route("/chat")
def chat():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def get_response():
    data = request.get_json()  # Parse the JSON data from the request
    user_query = data.get("query")
    # Retrieve the session's chat history or initialize it if not exists
    if "chat_history" not in session:
        session["chat_history"] = []

    # Append user's query to chat history
    session["chat_history"].append({"role": "user", "content": user_query})
    
    factual_response = generate_query(user_query)
    
    # Append the bot's response to chat history
    session["chat_history"].append({"role": "bot", "content": factual_response})
    return jsonify({"response": factual_response})

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    # Clear the chat history in the session
    session.pop("chat_history", None)
    return jsonify({"message": "Chat history cleared."})

# --- Main entry point ---
if __name__ == '__main__':
    create_vectordb()
    app.run(debug=True)
