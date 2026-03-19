import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from pydantic import BaseModel, Field

import pdfLoader
f''

PROMPT_TEMPLATE = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a biography reader that will answer questions only using the supplied context.
.
Keep the answer concise and within 2 sentences.

Context: {context}
""".strip(),
        ),
        ("human", "My query is: {question}"),
    ]
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Question for the chatbot")
    top_k: int = Field(1, ge=1, le=10, description="Number of context chunks")


class ChatResponse(BaseModel):
    answer: str
    context_used: list[str]


class IngestPdfRequest(BaseModel):
    path: str = Field(..., min_length=1, description="Path to PDF file")
    chunk_size: int = Field(1000, ge=100, le=5000)
    chunk_overlap: int = Field(200, ge=0, le=1000)


class IngestPdfResponse(BaseModel):
    chunks_indexed: int


class ChatbotService:
    def __init__(self) -> None:
        self.vector_store: PineconeVectorStore | None = None
        self.model: ChatGoogleGenerativeAI | None = None

    def initialize(self) -> None:
        load_dotenv()
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX", "prac-chat-bot")

        if not pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")
        if not google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")

        pc = Pinecone(api_key=pinecone_api_key)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.vector_store = PineconeVectorStore(
            index=pc.Index(index_name),
            embedding=embeddings,
        )
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=1.0,
            max_retries=2,
        )

    def _ensure_ready(self) -> None:
        if self.vector_store is None or self.model is None:
            raise RuntimeError("Service not initialized")

    def chat(self, question: str, top_k: int) -> ChatResponse:
        self._ensure_ready()
        context_docs = self.vector_store.similarity_search(question, k=top_k)
        chain = PROMPT_TEMPLATE | self.model
        print(context_docs)
        response = chain.invoke({"question": question, "context": context_docs})
        context_used = [doc.page_content for doc in context_docs]
        return ChatResponse(answer=response.content, context_used=context_used)

    def ingest_pdf(self, path: str, chunk_size: int, chunk_overlap: int) -> int:
        self._ensure_ready()
        loader = pdfLoader.PDFLoader(path)
        text = loader.extract_text()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_text(text)
        docs = [Document(page_content=chunk) for chunk in chunks]
        self.vector_store.add_documents(docs)
        return len(docs)


chatbot_service = ChatbotService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    chatbot_service.initialize()
    yield


app = FastAPI(
    title="Chatbot API",
    description="FastAPI service for retrieval-augmented chatbot",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return chatbot_service.chat(question=request.question, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# @app.post("/ingest/pdf", response_model=IngestPdfResponse)
# async def ingest_pdf(request: IngestPdfRequest) -> IngestPdfResponse:
#     try:
#         chunks = chatbot_service.ingest_pdf(
#             path=request.path,
#             chunk_size=request.chunk_size,
#             chunk_overlap=request.chunk_overlap,
#         )
#         return IngestPdfResponse(chunks_indexed=chunks)
#     except FileNotFoundError as exc:
#         raise HTTPException(status_code=404, detail="PDF file not found") from exc
#     except Exception as exc:
#         raise HTTPException(status_code=500, detail=str(exc)) from exc