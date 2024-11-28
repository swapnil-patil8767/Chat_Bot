import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.retrievers import MultiQueryRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Define constants
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Set up custom prompt template
custom_prompt_template = """You are a helpful and knowledgeable medical assistant. Answer the user's health-related questions by providing accurate, evidence-based information. Be compassionate and respectful in your responses, and remember that you are not a substitute for a licensed healthcare provider.

Only answer questions related to general medical information, lifestyle tips, symptoms of common conditions, or guidance on when to seek professional care. If the user asks for a diagnosis or complex medical advice, politely remind them to consult a qualified healthcare provider.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    # Remove chat_history from input variables since we're not using it in the template
    prompt = PromptTemplate(template=custom_prompt_template, 
                          input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    retriever = MultiQueryRetriever.from_llm(llm=llm, retriever=db.as_retriever())
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

def qa_bot():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def get_response(query: str):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response['result']

# Streamlit app configuration
st.set_page_config(page_title="Medical Chatbot 🏥", page_icon="🏥", layout="wide")

st.title("Medical Chatbot Assistant 🏥")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="👋 Hello! I'm your medical assistant. I can help you with general health questions and information. Please remember that I'm not a substitute for professional medical advice. How can I help you today?")
    ]

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="🏥"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="👤"):
            st.markdown(message.content)

# Chat input
user_query = st.chat_input("Type your health-related question here...")

if user_query is not None and user_query.strip() != "":
    # Add user message to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    # Display user message
    with st.chat_message("Human", avatar="👤"):
        st.markdown(user_query)
    
    # Get and display bot response
    with st.chat_message("AI", avatar="🏥"):
        with st.spinner("Thinking..."):
            response = get_response(user_query)
            st.markdown(response)
    
    # Add bot response to chat history
    st.session_state.chat_history.append(AIMessage(content=response))

# Add a footer
st.markdown("""
---
<div style='text-align: center'>
    <small>
        Remember: This chatbot is for informational purposes only. 
        Always consult with a healthcare provider for medical advice.
    </small>
</div>
""", unsafe_allow_html=True)