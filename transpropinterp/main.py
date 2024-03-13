import circuitsvis as cv
from transformer_lens import loading_from_pretrained
from transformer_lens import HookedTransformer
import torch

WELTERWEIGHT_FT = "sachalmalick/gpt2-transprop-ft-welterweight"
LIGHTWEIGHT_FT = "sachalmalick/gpt2-transprop-ft-lightweight"
FEATHERWEIGHT_FT = "sachalmalick/gpt2-transprop-ft-featherweight"

def register_finetuned_models():
    loading_from_pretrained.OFFICIAL_MODEL_NAMES.append(WELTERWEIGHT_FT)
    loading_from_pretrained.OFFICIAL_MODEL_NAMES.append(LIGHTWEIGHT_FT)
    loading_from_pretrained.OFFICIAL_MODEL_NAMES.append(FEATHERWEIGHT_FT)


def prompt_with_cache(model, text):
    tokens = model.to_tokens(text)
    logits, cache = model.run_with_cache(tokens)
    return logits, cache, tokens

def show_attention_patterns(tokens, layer, cache):
    attention_pattern = cache["pattern", layer, "attn"]
    cv.attention.attention_patterns(tokens=tokens, attention=attention_pattern)

def setup():
    register_finetuned_models()
    device = "cuda"
    model = HookedTransformer.from_pretrained(WELTERWEIGHT_FT, device=device)
    return model

def convert_strong_attenders_text(model, attenders, tokens):
    results = {}
    for head in attenders:
        string_pairs = []
        for pair_index in range(attenders[head].shape[0]):
           pair = attenders[head][pair_index]
           srctoken = tokens[pair[0]]
           distoken = tokens[pair[1]]
           string_pairs.append(model.to_string(srctoken).strip() + " " + str(pair[0].cpu().numpy()) + " -> " + model.to_string(distoken).strip() + " " + str(pair[1].cpu().numpy()))
        results[head] = set(string_pairs)
    return results


def get_strong_attenders(cache, layer, threshold=0.3, ignore_zero=True, ignore_end=False, ignore_first=True):
    #att_pttern = [num_heads, tokens, tokens]
    attention_pattern = cache["pattern", layer, "attn"].squeeze()
    num_heads = attention_pattern.shape[0]
    num_tokens = attention_pattern.shape[1]
    results = {}
    for i in range(num_heads):
        scores = attention_pattern[i]
        indices = torch.where(scores > threshold)
        row_indices = indices[0]
        col_indices = indices[1]
        index_pairs = torch.stack((row_indices, col_indices), dim=1)
        if(ignore_zero):
            index_pairs = index_pairs[index_pairs[:, 1] != 0]
            index_pairs = index_pairs[index_pairs[:, 0] != 0]
        if(ignore_first):
            index_pairs = index_pairs[index_pairs[:, 1] != 1]
            index_pairs = index_pairs[index_pairs[:, 0] != 1]
        if(ignore_end):
            index_pairs = index_pairs[index_pairs[:, 0] != num_tokens - 1]
            index_pairs = index_pairs[index_pairs[:, 1] != num_tokens - 1]
        if(index_pairs.shape[0] != 0):
            results[i] = index_pairs
    return results

def print_all_strong_attenders(model, cache, tokens):
    for layer in range(0, model.cfg.n_layers):
        print("============ Layer {} ============".format(layer))
        attenders = get_strong_attenders(cache, layer, threshold=0.3, ignore_zero=True)
        tokens = tokens.squeeze(0)
        text_strong_attenders = convert_strong_attenders_text(model, attenders, tokens)
        for i in text_strong_attenders:
            print("head ", i)
            for pair in sorted(list(text_strong_attenders[i])):
                print(pair)



if __name__ == "__main__":
    model = setup()
    logits, cache, tokens = prompt_with_cache(model, "if all men are humans and all humans suck it can be inferred that all men ")
    print_all_strong_attenders(model, cache, tokens)