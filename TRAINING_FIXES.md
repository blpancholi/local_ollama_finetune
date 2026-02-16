# Fixes for Fine-Tuning Issues

## Problem: Model not using fine-tuned content

### Issue 1: Insufficient Training
**Current:** `max_steps = 60` (only ~12 epochs on 39 samples)
**Fix:** Change training parameters in cell 5:

```python
# Replace this line:
max_steps = 60, # Set a limit for quick testing (e.g., 1 epoch)

# With this:
num_train_epochs = 30,  # Train for 30 epochs to ensure model learns
# max_steps = 60,  # Comment this out
```

**Why:** With only 39 samples, you need more epochs for the model to learn the patterns.

### Issue 2: Explicit Merge (Already Added)
A new cell has been added (cell 7) that:
- Explicitly merges LoRA adapters: `model = model.merge_and_unload()`
- Tests the merged model to verify it contains fine-tuned content

### Issue 3: Prompt Format Must Match
When calling the API, use the EXACT prompt format from training:

**Training format:**
```
Below is an instruction that describes a task, paired with an input that provides further context.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
```

**API Call Example:**
```json
{
  "model": "llama3.2-3b",
  "prompt": "Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction:\nPlease answer the following question\n\n### Input:\nGive me the company detail who provided the proposal?\n\n### Response:"
}
```

### Issue 4: Verify Training Loss
Check that training loss decreased significantly:
- Initial loss: ~3.0
- Final loss should be: < 0.1
- If loss didn't decrease much, training wasn't effective

## Recommended Training Parameters

```python
args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 10,
    num_train_epochs = 30,  # Increased for better learning
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = "outputs",
    report_to = "none",
    save_strategy = "epoch",  # Save after each epoch
    save_total_limit = 3,  # Keep only last 3 checkpoints
)
```

## Testing After Training

1. **Test in notebook** (cell 7) - Verify merge worked
2. **Test in Ollama** - Use exact prompt format
3. **Check response** - Should contain your fine-tuned content (e.g., "APN Solar")

## If Still Not Working

1. **Increase training data**: 39 samples is very small. Aim for 100+ samples
2. **Increase epochs**: Try 50+ epochs
3. **Check data quality**: Ensure training data is correct
4. **Verify model file**: Check that the GGUF file was created after training (check timestamp)

