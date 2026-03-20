import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient
import pdfLoader

PROMPT_TEMPLATE = """
You are a biography reader that will answer questions only using the supplied context.
Keep the answer concise and within 2 sentences.

Context: {context}

My query is: {question}
""".strip()


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


class HFInferenceEmbeddings:
    def __init__(self, client: InferenceClient, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> list[float]:
        vector = self.client.feature_extraction(
            text,
            model=self.model_name,
        )
        # HF can return a 1D array/list or a 2D array/list (batch with one row).
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if isinstance(vector, list) and vector and isinstance(vector[0], list):
            vector = vector[0]
        return [float(value) for value in vector]


class ChatbotService:
    def __init__(self) -> None:
        self.index = None
        self.genai_client: genai.Client | None = None
        self.embeddings: HFInferenceEmbeddings | None = None

    def initialize(self) -> None:
        load_dotenv()
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        hf_token = os.getenv("HF_TOKEN")
        index_name = os.getenv("PINECONE_INDEX", "prac-chat-bot")

        if not pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")
        if not google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        if not hf_token:
            raise RuntimeError("HF_TOKEN is not set")

        pc = Pinecone(api_key=pinecone_api_key)
        hf_client = InferenceClient(
            provider="hf-inference",
            api_key=hf_token,
        )
        embeddings = HFInferenceEmbeddings(
            client=hf_client,
            model_name="sentence-transformers/all-mpnet-base-v2",
        )
        self.index = pc.Index(index_name)
        self.embeddings = embeddings
        self.genai_client = genai.Client(api_key=google_api_key)

    def _ensure_ready(self) -> None:
        if self.index is None or self.genai_client is None or self.embeddings is None:
            raise RuntimeError("Service not initialized")

    def chat(self, question: str, top_k: int) -> ChatResponse:
        self._ensure_ready()
        query_vector = self.embeddings.embed_query(question)
        query_result = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )
        matches = query_result.get("matches", [])
        context_used = [
            match.get("metadata", {}).get("text", "")
            for match in matches
            if match.get("metadata", {}).get("text")
        ]
        context_text = "\n\n".join(context_used)
        prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
        response = self.genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return ChatResponse(answer=response.text or "", context_used=context_used)

    @staticmethod
    def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        chunks: list[str] = []
        start = 0
        step = chunk_size - chunk_overlap
        while start < len(text):
            chunk = text[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks

    def ingest_pdf(self, path: str, chunk_size: int, chunk_overlap: int) -> int:
        self._ensure_ready()
        loader = pdfLoader.PDFLoader(path)
        text = loader.extract_text()
        chunks = self._split_text(text, chunk_size, chunk_overlap)
        vectors = []
        for chunk in chunks:
            values = self.embeddings.embed_query(chunk)
            vectors.append(
                {
                    "id": str(uuid.uuid4()),
                    "values": values,
                    "metadata": {"text": chunk},
                }
            )

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i : i + batch_size])
        return len(vectors)


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