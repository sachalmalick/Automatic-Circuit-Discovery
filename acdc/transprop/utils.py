from collections import OrderedDict
from acdc.TLACDCEdge import (TorchIndex, Edge, EdgeType)
from acdc.TLACDCInterpNode import TLACDCInterpNode
import warnings
from functools import partial
from copy import deepcopy
import torch.nn.functional as F
from typing import List
from acdc.acdc_utils import kl_divergence
import torch
from acdc.ioi.ioi_dataset import IOIDataset
from tqdm import tqdm
import wandb
from transformer_lens.HookedTransformer import HookedTransformer
import warnings
from functools import partial
from typing import ClassVar, Optional

import torch
import torch.nn.functional as F

from acdc.acdc_utils import kl_divergence, TorchIndex
from acdc.docstring.utils import AllDataThings
from collections import OrderedDict
from acdc.acdc_utils import MatchNLLMetric, frac_correct_metric, logit_diff_metric, kl_divergence, negative_log_probs
from transformer_lens import loading_from_pretrained
import random



FIRST = [
    "cat", "dog", "house", "car", "tree", "bird", "table", "chair", "book", "computer",
    "phone", "flower", "pen", "pencil", "paper", "river", "mountain", "ocean", "lake", "sun",
    "moon", "star", "planet", "city", "country", "bicycle", "boat", "ship", "airplane", "train",
    "bus", "elephant", "lion", "tiger", "bear", "wolf", "fox", "rabbit", "horse", "snake", "fish",
    "apple", "banana", "orange", "grape", "strawberry", "watermelon", "melon", "pineapple", "pear",
    "peach", "plum", "kiwi", "mango", "cherry", "lemon", "lime", "coconut", "avocado", "potato",
    "tomato", "onion", "garlic", "lettuce", "cucumber", "carrot", "broccoli", "pepper", "mushroom",
    "eggplant", "corn", "spinach", "asparagus", "celery", "peas", "beans", "rice", "pasta", "bread",
    "cheese", "milk", "yogurt", "butter", "egg", "chicken", "beef", "pork", "fish", "shrimp", "lobster",
    "crab", "salad", "soup", "sandwich", "pizza", "burger", "fries", "cake", "cookie", "ice cream", "chocolate"
]

SECOND = [
    "desk", "lamp", "mirror", "bed", "window", "door", "key", "lock", "wallet", "bag",
    "umbrella", "jacket", "shirt", "pants", "socks", "shoes", "hat", "glasses", "watch", "bracelet",
    "ring", "necklace", "earrings", "television", "remote", "speaker", "microwave", "oven", "refrigerator",
    "sink", "toilet", "mirror", "brush", "comb", "shampoo", "soap", "towel", "laptop", "tablet", "keyboard",
    "mouse", "headphones", "charger", "battery", "clock", "calendar", "wallet", "coin", "bank", "wallet",
    "coin", "bank", "credit card", "debit card", "cash", "receipt", "ticket", "map", "compass", "camera",
    "binoculars", "telescope", "microscope", "guitar", "piano", "violin", "drums", "trumpet", "flute",
    "saxophone", "harmonica", "accordion", "banjo", "ukulele", "harp", "painting", "sculpture", "drawing",
    "photograph", "pottery", "vase", "statue", "figurine", "ornament", "artifact", "tapestry", "rug",
    "curtain", "pillow", "blanket", "quilt", "mattress", "candle", "incense", "perfume", "fragrance", "lotion"
]

THIRD = [
    "telephone", "mailbox", "backpack", "sunglasses", "umbrella", "raincoat", "wallet", "briefcase", "suitcase", "watch",
    "earbuds", "headphones", "headset", "thermometer", "thermostat", "umbrella", "canteen", "thermos", "snorkel", "swimsuit",
    "flippers", "raft", "lifejacket", "tent", "sleeping bag", "campfire", "binoculars", "backpack", "flashlight", "compass",
    "map", "treasure", "scroll", "treasure chest", "dagger", "sword", "shield", "armor", "crown", "throne",
    "wand", "staff", "potion", "scroll", "book", "spellbook", "amulet", "ring", "gemstone", "crystal",
    "bracelet", "necklace", "earrings", "candlestick", "vase", "statue", "painting", "portrait", "talisman", "scarab",
    "coin", "medallion", "artifact", "fountain", "gargoyle", "gate", "statue", "arch", "castle", "moat",
    "drawbridge", "dungeon", "tower", "fortress", "barricade", "rampart", "siege", "catapult", "ballista", "trebuchet",
    "battering ram", "ladder", "rope", "torch", "brazier", "cauldron", "grimoire", "alchemy", "potion", "elixir"
]

PATCH = "henry"

WELTERWEIGHT_FT = "sachalmalick/gpt2-transprop-ft-welterweight"
LIGHTWEIGHT_FT = "sachalmalick/gpt2-transprop-ft-lightweight"
FEATHERWEIGHT_FT = "sachalmalick/gpt2-transprop-ft-featherweight"

loading_from_pretrained.OFFICIAL_MODEL_NAMES.append(WELTERWEIGHT_FT)
loading_from_pretrained.OFFICIAL_MODEL_NAMES.append(LIGHTWEIGHT_FT)
loading_from_pretrained.OFFICIAL_MODEL_NAMES.append(FEATHERWEIGHT_FT)
    
def get_prompt_data(num_examples, model, power_template=None):
    if(power_template == None):
        power_template = "{a} implies {b} and if {b} then {c} therefore by the transitive property {a} also implies"
    random.seed(54)
    first = random.sample(FIRST, num_examples)
    second = random.sample(SECOND, num_examples)
    third = random.sample(THIRD, num_examples)
    prompts = [power_template.format(a=first[i], b=second[i], c=third[i]) for i in range(num_examples)]
    corrupt_prompts = prompts.copy()
    random.shuffle(corrupt_prompts)
    prompts_tokenized = model.tokenizer(prompts, padding=True, return_tensors="pt").to(model.cfg.device)
    corrupt_tokenized = model.tokenizer(corrupt_prompts, padding=True, return_tensors="pt").to(model.cfg.device)
    correct_labels = model.tokenizer([" " + i for i in third], padding=True, return_tensors="pt").to(model.cfg.device)
    wrong_labels = model.tokenizer([" " + i for i in second], padding=True, return_tensors="pt").to(model.cfg.device)
    return {"prompts": prompts, "first": first, "second": second, "third": third, "tokens": prompts_tokenized,
    "corrupt_prompts": corrupt_prompts, "corrupt_tokens": corrupt_tokenized, "correct_labels" : correct_labels, "wrong_labels" : wrong_labels}

def get_finetuned_gpt2(modelname=WELTERWEIGHT_FT,device="cuda"):
    tl_model = HookedTransformer.from_pretrained(modelname)
    tl_model = tl_model.to(device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)
    return tl_model

def get_transprop_things(num_examples,
                               metric_name,
                               device="cuda",
                               test_split=0.5,
                               model=None,
                               prompt=None,
                               kl_return_one_element=True):
    if(model == None):
        model = get_finetuned_gpt2(device=device)
    data = get_prompt_data(num_examples, model, power_template=prompt)
    prompt_patch_data = data["corrupt_tokens"]["input_ids"]
    prompt_data = data["tokens"]["input_ids"]
    train_examples = int(num_examples * test_split)
    validation_data = prompt_data[:train_examples]
    validation_patch_data = prompt_patch_data[:train_examples]
    test_data = prompt_data[train_examples:]
    test_patch_data = prompt_patch_data[train_examples:]
    validation_labels =  data["correct_labels"]["input_ids"][:train_examples][:,0]
    validation_wrong_labels =  data["wrong_labels"]["input_ids"][:train_examples][:,0]
    test_labels =  data["correct_labels"]["input_ids"][train_examples:][:,0]
    test_wrong_labels =  data["wrong_labels"]["input_ids"][train_examples:][:,0]

    validation_mask = data["tokens"]["attention_mask"][:train_examples]
    test_mask = data["tokens"]["attention_mask"][train_examples:]

    validation_indices = torch.sum(validation_mask, dim=1)
    test_indices = torch.sum(test_mask, dim=1)

    with torch.no_grad():
        base_model_logits = model(prompt_data)
        base_ind = torch.sum(data["tokens"]["attention_mask"], dim=1)
        base_model_logits = torch.stack([
            torch.squeeze(base_model_logits[i, base_ind[i] - 1, :]) for i in range(base_ind.shape[0])
        ])
        base_model_logprobs = F.log_softmax(base_model_logits, dim=-1)
        base_validation_logprobs = base_model_logprobs[:train_examples]
        base_test_logprobs = base_model_logprobs[train_examples:]
        print(base_validation_logprobs.shape)
    if metric_name == "kl_div":
        validation_metric = partial(
            kl_divergence,
            base_model_logprobs=base_validation_logprobs,
            last_seq_element_only=True,
            base_model_probs_last_seq_element_only=False,
            return_one_element=kl_return_one_element,
            padded=True,
            indices=validation_indices
        )
    elif metric_name == "logit_diff":
        validation_metric = partial(
            logit_diff_metric,
            correct_labels=validation_labels,
            wrong_labels=validation_wrong_labels,
        )
    elif metric_name == "frac_correct":
        validation_metric = partial(
            frac_correct_metric,
            correct_labels=validation_labels,
            wrong_labels=validation_wrong_labels,
        )
    elif metric_name == "nll":
        validation_metric = partial(
            negative_log_probs,
            labels=validation_labels,
            last_seq_element_only=True,
        )
    elif metric_name == "match_nll":
        validation_metric = MatchNLLMetric(
            labels=validation_labels,
            base_model_logprobs=base_validation_logprobs,
            last_seq_element_only=True,
        )
    else:
        raise ValueError(f"metric_name {metric_name} not recognized")

    test_metrics = {
        "kl_div": partial(
            kl_divergence,
            base_model_logprobs=base_test_logprobs,
            last_seq_element_only=True,
            base_model_probs_last_seq_element_only=False,
            padded=True,
            indices=test_indices
        ),
        "logit_diff": partial(
            logit_diff_metric,
            correct_labels=test_labels,
            wrong_labels=test_wrong_labels,
        ),
        "frac_correct": partial(
            frac_correct_metric,
            correct_labels=test_labels,
            wrong_labels=test_wrong_labels,
        ),
        "nll": partial(
            negative_log_probs,
            labels=test_labels,
            last_seq_element_only=True,
        ),
    }


    return AllDataThings(
        tl_model=model,
        validation_metric=validation_metric,
        validation_data=validation_data,
        validation_labels=validation_labels,
        validation_wrong_labels=validation_wrong_labels,
        validation_mask=validation_mask,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=test_mask,
        test_patch_data=test_patch_data)
# %%
