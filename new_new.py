from flask import Flask, render_template, request, jsonify
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Flask app setup ---
app = Flask(__name__)

# --- Load the vectorstore and embeddings ---
DB_FAISS_PATH = 'vectorstore/'  # Path to your vectorstore folder
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Load FAISS vectorstore (allow_dangerous_deserialization is required for local)
vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# Initialize retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Define LLM using Google Generative AI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=None)

# --- Prompt Template ---
prompt_template = """You are a helpful and honest chatbot. Use the following information to answer factually correct answers for user's query.
If you don't know the answer, clearly say you don't know the answer.

Context: {context}
Question: {question}

Only return the correct answer below and nothing else.
Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# --- Function to generate response based on query ---
def generate_query(query):
    # Create the QA chain to generate an answer
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": PROMPT})
    
    # Get response from the QA chain
    response = qa({"query": query})
    
    # Extract and return the result
    return response['result']


# --- Flask Routes ---
@app.route("/")
def home():
    return render_template('landing.html')

@app.route("/chat")
def chat():
    return render_template('chat.html')

# @app.route("/get", methods=["POST"])
# def get_response():
#     data = request.get_json()  # Parse the JSON data from the request
#     user_query = data.get("query")
    
#     # Generate a response using the RAG model
#     factual_response = generate_query(user_query)
    
#     # Return the response as JSON
#     return jsonify({"response": factual_response})

@app.route("/get", methods=["GET", "POST"])
def get_response():
    msg = request.form["msg"]
    
    # First try with retrieval-based factual chatbot model
    factual_response = generate_query(msg)
    
   
    
    # Return the factual response if available
    return factual_response

# --- Main entry point ---
if __name__ == '__main__':
    app.run(debug=True)
