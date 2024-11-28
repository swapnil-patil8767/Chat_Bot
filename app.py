from flask import Flask, request, jsonify
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.retrievers import MultiQueryRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define constants
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Custom prompt template
custom_prompt_template = """You are a helpful and knowledgeable medical assistant. Answer the user's health-related questions by providing accurate, evidence-based information. Be compassionate and respectful in your responses, and remember that you are not a substitute for a licensed healthcare provider.

Only answer questions related to general medical information, lifestyle tips, symptoms of common conditions, or guidance on when to seek professional care. If the user asks for a diagnosis or complex medical advice, politely remind them to consult a qualified healthcare provider.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    retriever = MultiQueryRetriever.from_llm(
        llm=llm,
        retriever=db.as_retriever()
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def load_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-70b-versatile"
    )
    return llm

# Initialize the QA bot
def initialize_qa_bot():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Initialize the QA bot at startup
qa_bot = initialize_qa_bot()

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Get response from QA bot
        response = qa_bot({'query': user_message})
        
        return jsonify({
            'response': response['result'],
            'success': True
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True)
