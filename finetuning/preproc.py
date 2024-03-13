import pickle
import nltk
from nltk.corpus import wordnet as wn
import random
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from util import pickle_obj, load_pickled_data
import torch

ALL_NOUNS_FILE = "data/all_nouns.pkl"
RANDOM_NOUN_PROMPTS = "data/random_noun_prompts.pkl"
RANDOM_NOUN_TOKENS = "data/random_noun_tokens.pkl"
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

#lets aim for 20,000 examples.

PROMPT_TEMPLATES = [
    "{a} implies {b} and if {b} then {c} therefore by the transitive property {a} also implies",
    "if all {a} are {b} and {b} are {c} then all {a} are",
    "if all {a} are {b} and {b} is a type of {c} then it can be inferred that all {a} are a type of",
    "{a} implies {b} and if {b} then {c} therefore {a} also implies",
    "{a} implies {b} and if {b} then {c} therefore by the transitive property {a} also implies",
    "{a} implies {b} and {b} implies {c} then by the transitive property {a} also implies",
    "{a} is a type of {b} and all {b} are {c} therefore {a} is also a type of"
]

class RandomNounPrompts(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, index):
        return {
            "attention_mask" : self.tokens[index]["attention_mask"],
            "input_ids": self.tokens[index]["input_ids"],
            "labels": self.tokens[index]["labels"]
        }

def generate_and_save_nounset(filepath, num_nouns):
    print("downloading worednet")
    nltk.download('wordnet')
    all_nouns = list(wn.all_synsets(pos=wn.NOUN))
    print("Have downloaded {} nouns".format(len(all_nouns)))
    if(num_nouns > len(all_nouns)):
        print("num nounds needed {} greater than num nouns {} so defaulting to num nouns".format(num_nouns, len(all_nouns)))
        num_nouns = len(all_nouns)
    selected_nouns = random.sample(all_nouns, num_nouns)
    selected_nouns = list(set([noun.lemmas()[0].name() for noun in selected_nouns]))
    samples_per_prompt = len(selected_nouns) // 3
    print("{} unique nouns, so {} per prompt".format(len(selected_nouns), samples_per_prompt))
    a_nouns = selected_nouns[:samples_per_prompt]
    b_nouns = selected_nouns[samples_per_prompt:samples_per_prompt*2]
    c_nouns = selected_nouns[samples_per_prompt*2:]
    nouns = {"a" : a_nouns, "b" : b_nouns, "c" : c_nouns}
    pickle_obj(filepath, nouns)
    print("Saved all nouns to {}".format(filepath))
    
def setup():
    generate_and_save_nounset(ALL_NOUNS_FILE, 90000)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    nounset = load_pickled_data(ALL_NOUNS_FILE)
    generate_and_save_prompts(nounset, tokenizer)
    generate_and_save_prompt_tokens()

def evaluate_nounset():
    nounset = load_pickled_data(ALL_NOUNS_FILE)
    print(len(nounset["a"]), len(nounset["b"]), len(nounset["c"]))
    all_unique_nouns = []
    all_unique_nouns.extend(nounset["a"])
    all_unique_nouns.extend(nounset["b"])
    all_unique_nouns.extend(nounset["c"])
    print("Valid unique dataset", len(set(all_unique_nouns)) == (len(nounset["a"]) + len(nounset["b"]) + len(nounset["c"])))

# def create_tokens_and_labels(tokenizer, prompt, c, max_length, pad_val):
#     prompt_tokens = tokenizer(prompt)
#     c_tokens = tokenizer(c)
#     pad_amount = max_length - (len(prompt_tokens["input_ids"]) + len(c_tokens["input_ids"]))
#     combined_tokens = torch.cat((torch.tensor(prompt_tokens["input_ids"]), torch.tensor(c_tokens["input_ids"]), torch.full((pad_amount,), pad_val)))
#     attention_zeros = torch.zeros(pad_amount)
#     combined_attention = torch.cat((torch.tensor(prompt_tokens["attention_mask"]), torch.tensor(c_tokens["attention_mask"]),attention_zeros))
#     labels = torch.cat((torch.full((len(prompt_tokens["input_ids"]),), -100), torch.tensor(c_tokens["input_ids"]), torch.full((pad_amount,), -100)))
#     return {"input_ids" : combined_tokens, "attention_mask": combined_attention, "labels" : labels}


def create_tokens_and_labels(tokenizer, prompt, c, max_length, pad_val):
    prompt_tokens = tokenizer(prompt)
    c_tokens = tokenizer(c)
    pad_amount = max_length - len(prompt_tokens["input_ids"])
    combined_tokens = torch.cat((torch.tensor(prompt_tokens["input_ids"]), torch.full((pad_amount,), pad_val)))
    attention_zeros = torch.zeros(pad_amount)
    combined_attention = torch.cat((torch.tensor(prompt_tokens["attention_mask"]) , attention_zeros))
    label_pad = max_length - (len(prompt_tokens["input_ids"]) + len(c_tokens["input_ids"]))
    labels = torch.cat((torch.full((len(prompt_tokens["input_ids"]),), -100), torch.tensor(c_tokens["input_ids"]), torch.full((label_pad,), -100)))
    return {"input_ids" : combined_tokens, "attention_mask": combined_attention, "labels" : labels}



def generate_and_save_prompts(nounset, tokenizer):
    prompts = []
    total_possible_prompts = len(nounset["a"])
    for i in range(0, total_possible_prompts):
        for prompt_template in PROMPT_TEMPLATES:
            a = nounset["a"][i]
            b = nounset["b"][i]
            c = nounset["c"][i]
            prompt = prompt_template.format(a=nounset["a"][i], b=nounset["b"][i], c=nounset["c"][i])
            text = prompt + " " + c
            tokens = tokenizer(c)
            prompts.append({"prompt" : prompt, "a" : a, "b" : b, "c" : c, "text" : text, "input_ids" : tokens["input_ids"], "attention_mask" : tokens["attention_mask"]})
    print("generated {} unique prompts".format(len(prompts)))
    pickle_obj(RANDOM_NOUN_PROMPTS, prompts)
    print("Saved prompts to ", RANDOM_NOUN_PROMPTS)

def generate_and_save_prompt_tokens():
    prompts = load_pickled_data(RANDOM_NOUN_PROMPTS)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    all_tokens = []
    pad_val = 50256
    max_length = 100
    print("tokenizing all prompts")
    for i in prompts:
        all_tokens.append(create_tokens_and_labels(tokenizer, i["prompt"] +  " ", i["c"], max_length, pad_val))
    print("done tokenizing all prompts")
    pickle_obj(RANDOM_NOUN_TOKENS, all_tokens)
    print("Saved prompts to ", RANDOM_NOUN_TOKENS)

def get_random_nouns_ds():
    print("loading tokens")
    tokens = load_pickled_data(RANDOM_NOUN_TOKENS)
    print("done loading tokens")
    return RandomNounPrompts(tokens)

def tokenize_dataset(examples):
    return tokenizer(examples["text"])


if __name__ == "__main__":
    #setup()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # nounset = load_pickled_data(ALL_NOUNS_FILE)
    # generate_and_save_prompts(nounset, tokenizer)
    #generate_and_save_prompt_tokens()
    
    prompt_ds = get_random_nouns_ds()
    print(len(prompt_ds))
    print(prompt_ds[23])

    max_length = len(prompt_ds[23]["input_ids"])
    prompts = load_pickled_data(RANDOM_NOUN_PROMPTS)

    print("original tokens length", len(prompt_ds[23]["input_ids"]))

    pad_val = 50256
    prompt = prompts[23]["prompt"] + " "
    c = prompts[23]["c"]
    t_and_l = create_tokens_and_labels(tokenizer, prompt, c, max_length, pad_val)
    print(prompt_ds[23])

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    result = model(prompt_ds[23]["input_ids"], labels=prompt_ds[23]["labels"])
    print(result[0])


    # print("original tokens length", len(prompt_ds[23]["input_ids"]))
    # print("new tokens length", t_and_l["input_ids"].shape)

    # print("original attention_mask length", len(prompt_ds[23]["attention_mask"]))
    # print("new attention_mask length", t_and_l["attention_mask"].shape)

    # print("original attention_mask sum", sum(prompt_ds[23]["attention_mask"]))
    # print("new attention_mask sum", torch.sum(t_and_l["attention_mask"]))

    # print("new labels length", t_and_l["labels"].shape)

    # print(tokenizer.decode(prompt_ds[23]["input_ids"]))
    # print(tokenizer.decode(t_and_l["input_ids"]))
    # print(tokenizer.decode([15262,274,6,62,1169,29625]))