# Ollama Fine-Tuning and Deployment Guide

This project demonstrates how to fine-tune a Llama 3.2-3B model using LoRA adapters, convert it to GGUF format, and deploy it using Ollama with Docker and a FastAPI service.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Install and Configure Ollama](#step-1-install-and-configure-ollama)
3. [Step 2: Generate Training Data](#step-2-generate-training-data) [This can be automate with api]
4. [Step 3: Fine-Tune Model on Google Colab](#step-3-fine-tune-model-on-google-colab)
5. [Step 4: Download and Prepare GGUF File](#step-4-download-and-prepare-gguf-file)
6. [Step 5: Deploy with Docker](#step-5-deploy-with-docker)
7. [Step 6: Configure Environment Variables](#step-6-configure-environment-variables)
8. [Step 7: Start the Python Service](#step-7-start-the-python-service)
9. [Step 8: Test the API](#step-8-test-the-api)
10. [Troubleshooting](#troubleshooting)
11. [Project Structure](#project-structure)

---

## Prerequisites

- Docker and Docker Compose installed
- Python 3.12+ (for local development)
- Google Colab account (for fine-tuning)
- ChatGPT Plus or API access (for generating training data)
- Postman or curl (for API testing)

---

## Step 1: Install and Configure Ollama

### Option A: Install Ollama on Docker (Recommended)

1. **Clone or navigate to the project directory:**
   ```bash
   cd /path/to/ollama_test
   ```

2. **Review the `docker-compose.yml` file:**
   The file is already configured with:
   - Ollama service on port `11434`
   - Volume mounts for models and workspace
   - AI service on port `8080`

3. **Start Docker containers:**
   ```bash
   docker-compose up -d
   ```

4. **Verify Ollama is running:**
   ```bash
   docker ps
   ```
   You should see both `ollama` and `ai-service` containers running.

5. **Test Ollama directly:**
   ```bash
   docker exec ollama ollama list
   ```

### Option B: Install Ollama on Windows

1. **Download Ollama:**
   - Visit [https://ollama.ai/download](https://ollama.ai/download)
   - Download the Windows installer

2. **Install Ollama:**
   - Run the installer and follow the setup wizard
   - Ollama will be available as a service

3. **Verify installation:**
   ```powershell
   ollama --version
   ```

4. **Test Ollama:**
   ```powershell
   curl http://localhost:11434/api/tags
   ```

---

## Step 2: Generate Training Data

### 2.1 Upload PDF to ChatGPT
### NOTE This can be automate with api call and increase number of question by recursive approach for each question.


1. **Open ChatGPT** (ChatGPT Plus or API)

2. **Upload your PDF document** containing the content you want to fine-tune on

3. **Request JSON generation:**
   Use the following prompt:
   ```
   Please analyze this PDF and generate a comprehensive set of question-answer pairs 
   in JSON format. The JSON should follow this structure:
   
   [
     {
       "instruction": "Explain the following content clearly and accurately.",
       "input": "Your question here",
       "output": "Your answer here"
     }
   ]
   
   Generate at least 100 diverse question-answer pairs covering all important 
   topics in the document.
   ```

### 2.2 Save the JSON File

1. **Copy the generated JSON** from ChatGPT
2. **Save it as `solar_qa.json`** (or your preferred name) in the project root
3. **Verify the JSON format** - it should match the structure shown above

**Example JSON structure:**
```json
[
  {
    "instruction": "Explain the following content clearly and accurately.",
    "input": "What is the purpose of the solar project?",
    "output": "The project aims to install a 10KW on-grid rooftop solar power system..."
  }
]
```

---

## Step 3: Fine-Tune Model on Google Colab

### 3.1 Upload JSON to Google Colab

1. **Open Google Colab** and create a new notebook
2. **Upload your JSON file** (`solar_qa.json`) to Colab:
   - Click the folder icon in the left sidebar
   - Click "Upload" and select your JSON file

### 3.2 Run the Fine-Tuning Notebook

1. **Open the notebook:** `finetune_with_adapter_training_working.ipynb`

2. **Execute cells in order:**
   
   **Cell 1: Install dependencies**
   ```python
   !pip install unsloth trl peft accelerate bitsandbytes pandas psutil psutils
   !pip install "trl<0.15.0" psutil --upgrade
   ```

   **Cell 2: Load model**
   ```python
   from unsloth import FastLanguageModel
   import torch
   
   model_name = "unsloth/llama-3.2-3b-bnb-4bit"
   max_seq_length = 2048
   dtype = None
   
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name=model_name,
       max_seq_length=max_seq_length,
       dtype=dtype,
       load_in_4bit=True,
   )
   ```

   **Cell 3: Add LoRA adapters**
   ```python
   model = FastLanguageModel.get_peft_model(
       model,
       r=64,
       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
       lora_alpha=128,
       lora_dropout=0,
       bias="none",
       use_gradient_checkpointing="unsloth",
       random_state=3407,
       use_rslora=False,
       loftq_config=None,
   )
   ```

   **Cell 4: Load and prepare dataset**
   ```python
   import json
   from datasets import Dataset
   
   json_file_path = "solar_qa.json"
   with open(json_file_path, 'r', encoding='utf-8') as f:
       data = json.load(f)
   
   # Convert to dataset format (see notebook for full code)
   ```

   **Cell 5: Train the model**
   ```python
   from trl import SFTTrainer
   from transformers import TrainingArguments
   
   trainer = SFTTrainer(
       model=model,
       tokenizer=tokenizer,
       train_dataset=dataset,
       dataset_text_field="text",
       max_seq_length=max_seq_length,
       args=TrainingArguments(
           per_device_train_batch_size=2,
           gradient_accumulation_steps=4,
           warmup_steps=10,
           num_train_epochs=50,
           learning_rate=2e-4,
           fp16=not torch.cuda.is_bf16_supported(),
           bf16=torch.cuda.is_bf16_supported(),
           logging_steps=1,
           optim="adamw_8bit",
           weight_decay=0.01,
           lr_scheduler_type="cosine",
           seed=3407,
           output_dir="outputs",
           report_to="none",
       ),
   )
   
   trainer.train()
   ```

   **Cell 6: Test the model** (optional but recommended)
   - Run inference cells to verify the model works correctly

   **Cell 7: Export to GGUF**
   ```python
   model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")
   ```
   
   **Note:** `q4_k_m` provides a balanced quality/speed tradeoff (~2GB). For better quality, use `Q8_0` (~3GB).

### 3.3 Download GGUF File

1. **After training completes**, locate the GGUF file in the `gguf_model` folder
2. **Download the file** (usually named `Llama-3.2-3B.Q4_K_M.gguf`)
   - Right-click the file in Colab's file browser
   - Select "Download"

---

## Step 4: Download and Prepare GGUF File

1. **Place the GGUF file** in the project root directory:
   ```bash
   /path/to/ollama_test/Llama-3.2-3B.Q4_K_M.gguf
   ```

2. **Verify the file exists:**
   ```bash
   ls -lh Llama-3.2-3B.Q4_K_M.gguf
   ```

3. **Ensure the `Modelfile` is configured correctly:**
   ```dockerfile
   FROM /workspace/Llama-3.2-3B.Q4_K_M.gguf
   
   PARAMETER temperature 0.7
   PARAMETER top_p 0.9
   PARAMETER top_k 40
   ```

---

## Step 5: Deploy with Docker

### 5.1 Start Docker Containers

1. **Navigate to project directory:**
   ```bash
   cd /path/to/ollama_test
   ```

2. **Stop any running containers:**
   ```bash
   docker-compose down
   ```

3. **Start Docker containers:**
   ```bash
   docker-compose up -d
   ```

4. **Verify containers are running:**
   ```bash
   docker ps
   ```

### 5.2 Create Model in Ollama

1. **Create the model using the Modelfile:**
   ```bash
   docker exec ollama ollama create llama3.2-3b -f /workspace/Modelfile
   ```

2. **Verify the model was created:**
   ```bash
   docker exec ollama ollama list
   ```
   You should see `llama3.2-3b` in the list.

### 5.3 Test Ollama Directly

**Option 1: Using Docker exec**
```bash
docker exec ollama ollama run llama3.2-3b "Tell me about APN company?"
```

**Option 2: Using API endpoint**
```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2-3b",
    "prompt": "What type of inverter is proposed for the system?",
    "stream": false
  }'
```

**Option 3: Using Postman**
- **Method:** POST
- **URL:** `http://localhost:11434/api/generate`
- **Headers:** `Content-Type: application/json`
- **Body:**
  ```json
  {
    "model": "llama3.2-3b",
    "prompt": "What type of inverter is proposed for the system?",
    "stream": false
  }
  ```

---

## Step 6: Configure Environment Variables

1. **Create a `.env` file** in the project root (if it doesn't exist):
   ```bash
   touch .env
   ```

2. **Add the following variables:**
   ```env
   OLLAMA_URL=http://localhost:11434
   MODEL_NAME=llama3.2-3b
   ```

3. **For Docker deployment**, the environment is already configured in `docker-compose.yml`:
   ```yaml
   environment:
     - OLLAMA_URL=http://ollama:11434
   ```

**Note:** The `main.py` file reads `OLLAMA_URL` from environment variables with a default fallback to `http://localhost:11434`.

---

## Step 7: Start the Python Service

### Option A: Using Docker (Recommended)

The service is already configured in `docker-compose.yml` and starts automatically with:
```bash
docker-compose up -d
```

**Check service logs:**
```bash
docker logs ai-service
```

**Check if service is running:**
```bash
docker ps | grep ai-service
```

### Option B: Local Development

1. **Create a virtual environment:**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables:**
   ```bash
   export OLLAMA_URL=http://localhost:11434  # On Windows: set OLLAMA_URL=http://localhost:11434
   ```

4. **Start the service:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080
   ```

5. **Verify the service is running:**
   ```bash
   curl http://localhost:8080/health
   ```

---

## Step 8: Test the API

### 8.1 Health Check

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "ok",
  "ollama_url": "http://localhost:11434"
}
```

### 8.2 Test Generate Endpoint

**Using Postman:**

1. **Method:** POST
2. **URL:** `http://localhost:8080/generate` (or `http://localhost:8001/generate` if you've configured a different port)
3. **Headers:** `Content-Type: application/json`
4. **Body:**
   ```json
   {
     "model": "llama3.2-3b",
     "prompt": "What type of inverter is proposed for the system?"
   }
   ```

**Using curl:**
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2-3b",
    "prompt": "What type of inverter is proposed for the system?"
  }'
```

**Expected response:**
```json
{
  "response": "A 10KW, three-phase solar inverter from V Sole, K Solar, or an equivalent manufacturer is proposed to ensure efficient power generation."
}
```

### 8.3 Test Chat Endpoint

**Using Postman:**

1. **Method:** POST
2. **URL:** `http://localhost:8080/chat`
3. **Headers:** `Content-Type: application/json`
4. **Body:**
   ```json
   {
     "model": "llama3.2-3b",
     "messages": [
       {
         "role": "user",
         "content": "What warranty coverage is provided for the solar inverter?"
       }
     ]
   }
   ```

**Using curl:**
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2-3b",
    "messages": [
      {
        "role": "user",
        "content": "What warranty coverage is provided for the solar inverter?"
      }
    ]
  }'
```

### 8.4 Other Available Endpoints

**Summarize:**
```bash
curl -X POST http://localhost:8080/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2-3b",
    "text": "Your long text here..."
  }'
```

**Embedding:**
```bash
curl -X POST http://localhost:8080/embedding \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text",
    "text": "Your text here"
  }'
```

---

## Troubleshooting

### Issue: Docker containers won't start

**Solution:**
```bash
# Check Docker daemon is running
docker ps

# Check logs
docker-compose logs

# Restart Docker service
sudo systemctl restart docker  # Linux
# Or restart Docker Desktop on Windows/Mac
```

### Issue: Model not found in Ollama

**Solution:**
```bash
# Verify GGUF file exists in project root
ls -lh Llama-3.2-3B.Q4_K_M.gguf

# Recreate the model
docker exec ollama ollama rm llama3.2-3b
docker exec ollama ollama create llama3.2-3b -f /workspace/Modelfile

# Verify model exists
docker exec ollama ollama list
```

### Issue: Service can't connect to Ollama

**Solution:**
```bash
# Check if Ollama container is running
docker ps | grep ollama

# Check Ollama logs
docker logs ollama

# Verify Ollama is accessible
curl http://localhost:11434/api/tags

# For Docker service, ensure OLLAMA_URL=http://ollama:11434
# For local service, ensure OLLAMA_URL=http://localhost:11434
```

### Issue: Port already in use

**Solution:**
```bash
# Find process using port 8080
lsof -i :8080  # Mac/Linux
netstat -ano | findstr :8080  # Windows

# Kill the process or change port in docker-compose.yml
```

### Issue: GGUF file too large or training fails

**Solution:**
- Use `q4_k_m` quantization for smaller file size (~2GB)
- Ensure you have enough GPU memory in Colab (use T4 GPU)
- Reduce `num_train_epochs` if training takes too long
- Reduce `max_seq_length` if you run out of memory

### Issue: Model responses are not accurate

**Solution:**
- Increase training epochs (currently 50)
- Add more diverse training data
- Adjust learning rate (currently 2e-4)
- Use better quantization method (Q8_0 instead of q4_k_m)

---

## Project Structure

```
ollama_test/
├── README.md                          # This file
├── docker-compose.yml                 # Docker Compose configuration
├── Dockerfile                         # Python service Dockerfile
├── Modelfile                          # Ollama model configuration
├── requirements.txt                   # Python dependencies
├── main.py                            # FastAPI service
├── solar_qa.json                      # Training data (Q&A pairs)
├── Llama-3.2-3B.Q4_K_M.gguf          # Fine-tuned model (GGUF format)
├── finetune_with_adapter_training_working.ipynb  # Training notebook
├── update_model.sh                    # Script to update model in Docker
├── .env                               # Environment variables (create this)
├── models/                            # Ollama models directory (created by Docker)
├── outputs/                           # Training outputs (created during training)
└── venv/                              # Python virtual environment (create this)
```

---

## Quick Start Commands

```bash
# 1. Start Docker containers
docker-compose up -d

# 2. Create model in Ollama
docker exec ollama ollama create llama3.2-3b -f /workspace/Modelfile

# 3. Verify model exists
docker exec ollama ollama list

# 4. Test Ollama directly
docker exec ollama ollama run llama3.2-3b "What type of inverter is proposed?"

# 5. Test API service
curl http://localhost:8080/health

# 6. Test generate endpoint
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2-3b", "prompt": "What type of inverter is proposed?"}'
```

---

## Updating the Model

If you need to update the fine-tuned model:

1. **Place new GGUF file** in project root
2. **Stop containers:**
   ```bash
   docker-compose down
   ```
3. **Start containers:**
   ```bash
   docker-compose up -d
   ```
4. **Create new model:** [update the model name to some new name]
   ```bash
   docker exec ollama ollama create llama3.3-3b -f /workspace/Modelfile
   ```
5. **Remove old model:**
   ```bash
   docker exec ollama ollama rm llama3.2-3b
   ```
6. **Remove old model name from APP add new model_name:**
 * Update it from .env file , config file or some DB variable (so we can update runtime)

Or use the provided script:
```bash
./update_model.sh /path/to/new/gguf/file.gguf
```

---
Configure Ollama on windows server 

1. Install Ollama on Windows
    Download the Windows installer from: https://ollama.ai/download
    Run the installer and complete setup.
    Open Terminal / PowerShell and verify:
        ollama --version

    Make sure the Ollama service is running:
        ollama list
        (Should return an empty or small list without errors.)

2. Prepare the GGUF and Modelfile on Windows
    Assume you have downloaded your fine‑tuned file, e.g. Llama-3.2-3B.Q4_K_M.gguf.
    Create a working folder (for example):
        mkdir C:\ollama-custom-model   cd C:\ollama-custom-model
    Copy your GGUF file into this folder:
    "
        copy "C:\path\to\Llama-3.2-3B.Q4_K_M.gguf" "C:\ollama-custom-model\"
    Create a Modelfile in C:\ollama-custom-model with contents like:
        FROM ./Llama-3.2-3B.Q4_K_M.gguf   PARAMETER temperature 0.7   PARAMETER top_p 0.9   PARAMETER top_k 40
    > Note: using ./ here is correct because you will run ollama create from this folder.
3. Create and Run the Custom Model (First Time)
    From PowerShell in C:\ollama-custom-model:
    Create the model in Ollama:
        cd C:\ollama-custom-model   
        ollama create llama3.2-3b -f Modelfile

    Verify the model exists:
        ollama list
        You should see llama3.2-3b in the list.
    
    Test the model directly:
        ollama run llama3.2-3b "Tell me about APN company?"
    Test via HTTP API (optional check):
        '
        curl -X POST http://localhost:11434/api/generate `     -H "Content-Type: application/json" `     -d '{ "model": "llama3.2-3b", "prompt": "What type of inverter is proposed for the system?", "stream": false }'


---

## Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LoRA Fine-tuning Guide](https://huggingface.co/docs/peft/index)

---

## License

This project is for educational and development purposes.

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review Docker logs: `docker-compose logs`
3. Review service logs: `docker logs ai-service`
4. Verify all prerequisites are installed correctly

