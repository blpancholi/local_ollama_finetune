from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama
import os

app = FastAPI(title="AI Microservice (Ollama)")

# Read OLLAMA API URL from environment variable
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Configure the Ollama client
client = ollama.Client(host=OLLAMA_URL)

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request Models
# -----------------------------

class GenerateRequest(BaseModel):
    model: str = "llama3.1"
    prompt: str


class ChatRequest(BaseModel):
    model: str = "llama3.1"
    messages: list


class SummaryRequest(BaseModel):
    model: str = "llama3.1"
    text: str


class EmbeddingRequest(BaseModel):
    model: str = "nomic-embed-text"
    text: str


# -----------------------------
# Helper Functions
# -----------------------------

def format_finetuned_prompt(instruction: str, input_text: str = "") -> str:
    """
    Format prompt to match the training format used during fine-tuning.
    This ensures the fine-tuned model recognizes the prompt structure.
    """
    prompt = """Below is an instruction that describes a task, paired with an input that provides further context.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
""".format(instruction=instruction, input_text=input_text)
    return prompt


# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/health")
def health():
    return {"status": "ok", "ollama_url": OLLAMA_URL}


@app.post("/generate")
def generate_text(req: GenerateRequest):
    try:
        # If prompt doesn't already have the training format, format it
        # Check if prompt contains the training format markers
        if "### Instruction:" not in req.prompt:
            # Assume it's a simple question and format it
            formatted_prompt = format_finetuned_prompt(
            instruction="Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction:\nExplain the following content clearly and accurately.\n\n###",
                input_text=req.prompt
            )
        else:
            # Already formatted, use as-is
            formatted_prompt = req.prompt
            
        response = client.generate(
            model=req.model,
            prompt=formatted_prompt,
            stream=False
        )
        return {"response": response.get("response")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        # For fine-tuned models, use generate with formatted prompt instead of chat
        # Extract user message from messages list
        user_message = ""
        if req.messages and len(req.messages) > 0:
            # Get the last user message
            for msg in reversed(req.messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
        
        # Format prompt for fine-tuned model
        formatted_prompt = format_finetuned_prompt(
            instruction="Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction:\nExplain the following content clearly and accurately.\n\n###",
            input_text=user_message
        )
        
        # Use generate instead of chat for fine-tuned models
        response = client.generate(
            model=req.model,
            prompt=formatted_prompt,
            stream=False
        )
        return {"response": response.get("response", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
def summarize(req: SummaryRequest):
    try:
        prompt = f"Summarize the following text in concise bullet points:\n\n{req.text}"
        response = client.generate(
            model=req.model,
            prompt=prompt,
            stream=False
        )
        return {"summary": response["response"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embedding")
def create_embedding(req: EmbeddingRequest):
    try:
        response = client.embed(
            model=req.model,
            input=req.text
        )
        return {"embedding": response["embeddings"][0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
