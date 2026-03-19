import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# RAG components
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# ------------------ LOAD ENV ------------------
load_dotenv()

# ------------------ GEMINI MODEL ------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)

# ------------------ RAG SETUP ------------------

# Embedding model (FREE)
embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load your data
loader = TextLoader("data.txt")
documents = loader.load()

# Split into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# Create / load vector DB
vectorstore = Chroma.from_documents(
    docs,
    embedding,
    persist_directory="./chroma_db"
)

# Persist DB (so it doesn't rebuild every time)
vectorstore.persist()

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ------------------ CONTEXT ------------------
portfolio_context = """
You are a chatbot for Prakash Bhattarai's portfolio website.
Only answer questions about Prakash.
If the question is unrelated, say you don't know.
"""

# ------------------ FASTAPI ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://prakash.dev.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    # 1. Retrieve relevant docs
    relevant_docs = retriever.invoke(req.message)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 2. Create prompt
    messages = [
        SystemMessage(content=portfolio_context),
        HumanMessage(content=f"""
Context:
{context}

Question:
{req.message}
""")
    ]

    # 3. Generate response
    result = model.invoke(messages)

    return {
        "reply": result.content,
        "context_used": context  # optional (good for debugging/interviews)
    }

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pydantic import BaseModel
from fastapi import HTTPException

# ------------------ REQUEST MODEL ------------------
class ContactRequest(BaseModel):
    name: str
    email: str
    message: str

# ------------------ EMAIL ROUTE ------------------
@app.post("/contact")
async def send_email(req: ContactRequest):
    try:
        sender_email = os.getenv("EMAIL_USER")
        sender_password = os.getenv("EMAIL_PASS")

        if not sender_email or not sender_password:
            raise HTTPException(status_code=500, detail="Email credentials not set")

        # Basic validation
        if not req.name.strip() or not req.email.strip() or not req.message.strip():
            raise HTTPException(status_code=400, detail="All fields are required")

        # Create email
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = sender_email  # you receive it
        msg["Subject"] = f"🚀 New Portfolio Message from {req.name}"

        # 🔥 IMPORTANT (so you can reply directly to user)
        msg["Reply-To"] = req.email

        # HTML body
        body = f"""
        <h2>New Contact Request</h2>
        <p><strong>Name:</strong> {req.name}</p>
        <p><strong>Email:</strong> {req.email}</p>
        <p><strong>Message:</strong></p>
        <p>{req.message}</p>
        """

        msg.attach(MIMEText(body, "html"))

        # Send email using Gmail SMTP
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        return {
            "status": "success",
            "message": "Email sent successfully"
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        print("Email Error:", e)
        raise HTTPException(status_code=500, detail="Failed to send email")