import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from src.trie import MarisaTrie
from src.graph_constrained_decoding import GraphConstrainedDecoding
from src.utils.graph_utils import build_graph, dfs
from src.utils.utils import path_to_string as path_to_str

def main():
    triples = [
        ("Albert Einstein", "born_in", "Germany"),
        ("Albert Einstein", "developed", "Theory of Relativity"),
        ("Albert Einstein", "won", "Nobel Prize"),
        ("Germany", "located_in", "Europe"),
        ("Nobel Prize", "awarded_by", "Nobel Foundation"),
        ("Marie Curie", "born_in", "Poland"),
        ("Marie Curie", "won", "Nobel Prize"),
        ("Marie Curie", "discovered", "Radium"),
        ("Poland", "located_in", "Europe"),
    ]

    G = build_graph(triples, undirected=False)
    all_paths = dfs(G, ["Albert Einstein"], 2)
    path_strings = [path_to_str(p) for p in all_paths]

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    # Build trie
    tokenized_paths = tokenizer(path_strings, padding=False, add_special_tokens=False).input_ids
    tokenized_path_list = [ids + [tokenizer.eos_token_id] for ids in tokenized_paths]
    trie = MarisaTrie(tokenized_path_list, max_token_id=len(tokenizer) + 1)

    prompt = "Generate a valid KG reasoning path starting from Albert Einstein: "
    chat = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    cfg = GenerationConfig(max_new_tokens=30, do_sample=False, num_beams=3, num_return_sequences=3,
                           return_dict_in_generate=True)

    # --- WITHOUT constraints ---
    print("=== WITHOUT Graph-Constrained Decoding ===")
    res_free = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                               generation_config=cfg, pad_token_id=tokenizer.eos_token_id)
    for i, seq in enumerate(res_free.sequences):
        out = tokenizer.decode(seq[input_ids.shape[1]:], skip_special_tokens=True)
        valid = " [VALID]" if out.strip() in path_strings else ""
        print(f"  {i+1}: {out}{valid}")

    # --- WITH constraints ---
    print("\n=== WITH Graph-Constrained Decoding ===")
    gcr = GraphConstrainedDecoding(tokenizer, trie, start_token_ids=None, end_token_ids=None,
                                   enable_constrained_by_default=True)
    res_gcr = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                              generation_config=cfg,
                              prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,
                              pad_token_id=tokenizer.eos_token_id)
    for i, seq in enumerate(res_gcr.sequences):
        out = tokenizer.decode(seq[input_ids.shape[1]:], skip_special_tokens=True)
        valid = " [VALID]" if out.strip() in path_strings else ""
        print(f"  {i+1}: {out}{valid}")

    print(f"\nValid paths in KG: {path_strings}")

if __name__ == "__main__":
    main()
