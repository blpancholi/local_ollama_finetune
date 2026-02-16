#!/bin/bash

# Script to update retrained model in Ollama Docker
# Usage: ./update_model.sh [path_to_new_gguf_file]

set -e

PROJECT_DIR="/Users/blpancholi/innovatechs_work/dev/ollama_test"
MODEL_NAME="llama3.2-3b"
GGUF_FILE="Llama-3.2-3B.Q4_K_M.gguf"

echo "🔄 Updating retrained model in Ollama Docker..."
echo ""

# Step 1: Copy new GGUF file if provided
if [ -n "$1" ]; then
    echo "📁 Copying new GGUF file..."
    if [ -f "$1" ]; then
        cp "$1" "$PROJECT_DIR/$GGUF_FILE"
        echo "✓ File copied successfully"
    else
        echo "❌ Error: File not found: $1"
        exit 1
    fi
else
    echo "ℹ️  No file path provided. Assuming $GGUF_FILE is already in project root."
fi

# Step 2: Ensure containers are running
echo ""
echo "🐳 Starting Docker containers..."
cd "$PROJECT_DIR"
docker-compose up -d

# Wait for containers to be ready
echo "⏳ Waiting for containers to start..."
sleep 5

# Step 3: Delete old model
echo ""
echo "🗑️  Removing old model..."
docker exec ollama ollama rm "$MODEL_NAME" 2>/dev/null || echo "ℹ️  Model doesn't exist yet (this is OK)"

# Step 4: Create new model
echo ""
echo "✨ Creating model with new GGUF file..."
docker exec ollama ollama create "$MODEL_NAME" -f /workspace/Modelfile

# Step 5: Verify
echo ""
echo "✅ Verifying model..."
docker exec ollama ollama list

echo ""
echo "🎉 Model update complete!"
echo ""
echo "Test the model with:"
echo "  docker exec ollama ollama run llama3.2-3b 'Tell me about APN company?'"

