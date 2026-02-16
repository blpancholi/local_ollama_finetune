# Steps to Update Retrained Model in Ollama Docker

## Prerequisites
- New GGUF file: `Llama-3.2-3B.Q4_K_M.gguf` (from your retrained model)
- Docker containers running

## Step-by-Step Process

### Step 1: Copy New GGUF File to Project Root
```bash
# If your new GGUF file is in a different location (e.g., from Colab/notebook)
# Copy it to the project root, replacing the old file:
cp /path/to/new/Llama-3.2-3B.Q4_K_M.gguf /Users/blpancholi/innovatechs_work/dev/ollama_test/
```

### Step 2: Stop Containers (Optional but recommended)
```bash
cd /Users/blpancholi/innovatechs_work/dev/ollama_test
docker-compose down
```

### Step 3: Delete Old Model from Ollama
```bash
# Start containers if stopped
docker-compose up -d

# Wait a few seconds for containers to start
sleep 5

# Delete the old model
docker exec ollama ollama rm llama3.2-3b
```

### Step 4: Recreate Model with New GGUF File
```bash
# Create the model again using the Modelfile (which points to /workspace/Llama-3.2-3B.Q4_K_M.gguf)
docker exec ollama ollama create llama3.2-3b -f /workspace/Modelfile
```

### Step 5: Verify Model is Updated
```bash
# List models to confirm
docker exec ollama ollama list

# Test the model
docker exec ollama ollama run llama3.2-3b "Tell me about APN company?"
```

### Step 6: Restart AI Service (if needed)
```bash
docker-compose restart ai-service
```

## Quick One-Liner Script

You can also run this all at once:

```bash
cd /Users/blpancholi/innovatechs_work/dev/ollama_test && \
docker-compose up -d && \
sleep 5 && \
docker exec ollama ollama rm llama3.2-3b 2>/dev/null || true && \
docker exec ollama ollama create llama3.2-3b -f /workspace/Modelfile && \
docker exec ollama ollama list
```

## Troubleshooting

### If model deletion fails:
```bash
# Force remove if needed
docker exec ollama ollama rm llama3.2-3b:latest --force
```

### If file not found:
```bash
# Verify file exists in container
docker exec ollama ls -lh /workspace/Llama-3.2-3B.Q4_K_M.gguf
```

### If model creation fails:
```bash
# Check Modelfile content
docker exec ollama cat /workspace/Modelfile

# Verify GGUF file is valid
docker exec ollama file /workspace/Llama-3.2-3B.Q4_K_M.gguf
```

## Notes

- The Modelfile already points to `/workspace/Llama-3.2-3B.Q4_K_M.gguf`
- Since the root directory is mounted as `/workspace`, any file in your project root is accessible
- No need to rebuild containers - just replace the file and recreate the model
- The model name `llama3.2-3b` stays the same, so your API calls don't need to change

