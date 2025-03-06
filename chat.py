import torch
from gpt import *

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda")
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval()

settings, params = load_gpt2(
    model_size="124M", models_dir="gpt2"
)

load_weights_into_gpt(gpt, params)
gpt.to(device)

torch.manual_seed(123)

while True:
    print(50 * "===")
    print("Faça uma pergunta")
    query = input()

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(f"{query}", tokenizer).to(device),
        max_new_tokens=100,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )
    print("Resposta:\n", token_ids_to_text(token_ids, tokenizer))