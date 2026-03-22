import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

import resend

load_dotenv()

# ------------------ FASTAPI ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://prakash-dev.vercel.app", "http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ GEMINI MODEL ------------------
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# ------------------ TOOL ------------------
@tool
def send_email(name: str, email: str, message: str) -> str:
    """
    Send an email/contact message to Prakash.
    Use this when the user wants to get in touch or contact Prakash.
    Extract name, email, and message from the conversation.
    Ask the user for any missing fields before calling this tool.
    """
    if not all([name.strip(), email.strip(), message.strip()]):
        return "Error: name, email, and message are all required."

    resend.api_key = os.getenv("RESEND_API_KEY")
    your_email = os.getenv("YOUR_EMAIL")

    params: resend.Emails.SendParams = {
        "from": "Portfolio Contact <onboarding@resend.dev>",
        "to": [your_email],
        "reply_to": email,
        "subject": f"🚀 New Portfolio Message from {name}",
        "html": f"""
            <h2>New Contact Request</h2>
            <p><strong>Name:</strong> {name}</p>
            <p><strong>Email:</strong> {email}</p>
            <p><strong>Message:</strong></p>
            <p>{message}</p>
        """,
    }
    resend.Emails.send(params)
    return f"Email sent successfully from {name} ({email})."

model_with_tools = model.bind_tools([send_email])

# ------------------ RAG (LAZY INIT) ------------------
retriever = None

def get_retriever():
    global retriever
    if retriever is None:
        print("⚡ Initializing RAG...")
        embedding = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        loader = TextLoader("data.txt")
        docs = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())
        retriever = FAISS.from_documents(docs, embedding).as_retriever(search_kwargs={"k": 3})
    return retriever

# ------------------ CONTEXT ------------------
portfolio_context = """
You are a chatbot for Prakash Bhattarai's portfolio website.
Only answer questions about Prakash.
If the question is unrelated, say you don't know.
You have access to a send_email tool. Use it when the user clearly
wants to contact or send a message to Prakash. If any of name, email,
or message are missing, ask the user for them before calling the tool.
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

@app.get("/warmup")
async def warmup():
    get_retriever()
    return {"status": "warm"}

# -------- CHAT (RAG + Tool Calling) --------
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        retriever_instance = get_retriever()
        relevant_docs = retriever_instance.invoke(req.message)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        messages = [
            SystemMessage(content=portfolio_context),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{req.message}"),
        ]

        response = model_with_tools.invoke(messages)
        tool_calls = getattr(response, "tool_calls", [])

        if tool_calls:
            tool_call = tool_calls[0]
            tool_id = tool_call["id"]

            tool_result = send_email.invoke(tool_call["args"])

            messages.append(response)
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_id))

            final_response = model_with_tools.invoke(messages)
            reply = final_response.content
        else:
            reply = response.content

        return {"reply": reply, "context_used": context}

    except Exception as e:
        print("Chat Error:", e)
        raise HTTPException(status_code=500, detail="Chat failed")

# -------- CONTACT (direct API use) --------
@app.post("/contact")
async def contact(req: ContactRequest):
    try:
        if not req.name.strip() or not req.email.strip() or not req.message.strip():
            raise HTTPException(status_code=400, detail="All fields are required")

        result = send_email.invoke({
            "name": req.name,
            "email": req.email,
            "message": req.message
        })

        if result.startswith("Error"):
            raise HTTPException(status_code=400, detail=result)

        return {"status": "success", "message": result}

    except HTTPException:
        raise
    except Exception as e:
        print("Email Error:", e)
        raise HTTPException(status_code=500, detail="Failed to send email")