import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# RAG components
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Email
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ------------------ LOAD ENV ------------------
load_dotenv()

# ------------------ FASTAPI ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://prakash.dev.vercel.app","http://127.0.0.1:5173","http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ GEMINI MODEL ------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)

# ------------------ RAG (LAZY INIT) ------------------
retriever = None

def get_retriever():
    global retriever

    if retriever is None:
        print("⚡ Initializing RAG...")

        embedding = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        loader = TextLoader("data.txt")
        documents = loader.load()

        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(docs, embedding)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return retriever

# ------------------ CONTEXT ------------------
portfolio_context = """
You are a chatbot for Prakash Bhattarai's portfolio website.
Only answer questions about Prakash.
If the question is unrelated, say you don't know.
"""

# ------------------ REQUEST MODELS ------------------
class ChatRequest(BaseModel):
    message: str

class ContactRequest(BaseModel):
    name: str
    email: str
    message: str

# ------------------ ROUTES ------------------
@app.get("/")
def home():
    return {"message": "API is running 🚀"}

# -------- CHAT (RAG) --------
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        retriever_instance = get_retriever()

        # Retrieve docs
        relevant_docs = retriever_instance.invoke(req.message)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Prompt
        messages = [
            SystemMessage(content=portfolio_context),
            HumanMessage(content=f"""
Context:
{context}

Question:
{req.message}
""")
        ]

        result = model.invoke(messages)

        return {
            "reply": result.content,
            "context_used": context
        }

    except Exception as e:
        print("Chat Error:", e)
        raise HTTPException(status_code=500, detail="Chat failed")

# -------- EMAIL --------
@app.post("/contact")
async def send_email(req: ContactRequest):
    try:
        sender_email = os.getenv("EMAIL_USER")
        sender_password = os.getenv("EMAIL_PASS")

        if not sender_email or not sender_password:
            raise HTTPException(status_code=500, detail="Email credentials not set")

        if not req.name.strip() or not req.email.strip() or not req.message.strip():
            raise HTTPException(status_code=400, detail="All fields are required")

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = sender_email
        msg["Subject"] = f"🚀 New Portfolio Message from {req.name}"
        msg["Reply-To"] = req.email

        body = f"""
        <h2>New Contact Request</h2>
        <p><strong>Name:</strong> {req.name}</p>
        <p><strong>Email:</strong> {req.email}</p>
        <p><strong>Message:</strong></p>
        <p>{req.message}</p>
        """

        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        return {"status": "success", "message": "Email sent successfully"}

    except HTTPException as e:
        raise e

    except Exception as e:
        print("Email Error:", e)
        raise HTTPException(status_code=500, detail="Failed to send email")