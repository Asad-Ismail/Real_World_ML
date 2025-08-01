from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, GenerationConfig

model_name = "gpt2"
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Check if generation_config exists, and add it if missing
if not hasattr(model, 'generation_config'):
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    
print("Model initialized successfully!")
print(f"Model has generation_config: {hasattr(model, 'generation_config')}")