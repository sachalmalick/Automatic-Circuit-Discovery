import pickle
from transformers import PushToHubCallback
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_pickled_data(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)
    
def pickle_obj(filepath, obj):
    with open(filepath, 'wb') as file:
        pickle.dump(obj, file)

def push_model(savepath, modelname):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained(savepath)
    model.push_to_hub("sachalmalick/" + modelname)
    tokenizer.push_to_hub("sachalmalick/" + modelname)


def prompt(text, model):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))