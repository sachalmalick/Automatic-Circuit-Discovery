from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, random_split
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from preproc import get_random_nouns_ds
import inspect
from torch.utils.data import Subset
from transformers import DataCollatorForLanguageModeling
from util import pickle_obj, load_pickled_data
import torch

class CustomDataCollatorForLanguageModeling:
    def __init__(self, tokenizer: GPT2Tokenizer, mlm: bool = False):
        self.tokenizer = tokenizer
        self.mlm = mlm  # GPT-2 is a causal model, so this should be False.

    def __call__(self, examples):
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        return batch

def split_dataset(ds, train_perc=.5, eval_perc=.3, batch_size=64):
    total_size = len(ds)
    train_size = int(total_size * train_perc)
    val_size = int(total_size * eval_perc)
    test_size = total_size - train_size - val_size
    print("train_size", train_size, "test_size", test_size, "eval_size", val_size)
    train_dataset, test_dataset, val_dataset = random_split(ds, [train_size, test_size, val_size])
    return train_dataset, test_dataset, val_dataset



def huggingface_finetune(prompts_ds, save_name, train_perc=.5, eval_perc=.3, batch_size=64):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # gpt2 is causal
    )
    model = GPT2LMHeadModel.from_pretrained('gpt2')    
    train_loader, test_loader, eval_loader = split_dataset(prompts_ds, train_perc=train_perc, eval_perc=eval_perc, batch_size=batch_size)
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    training_args = TrainingArguments(output_dir="training", evaluation_strategy="epoch", learning_rate=5e-5, 
        label_names=["labels"], num_train_epochs=20)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader,
        eval_dataset=eval_loader,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    model.train()
    trainer.train()
    model.save_pretrained(save_name)
    pickle_obj("data/train2.pkl", train_loader)
    pickle_obj("data/test2.pkl", test_loader)
    pickle_obj("data/eval2.pkl", eval_loader)

def test_finetuned_model(path):
    original_model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained(path)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.encode("cat implies drink and if drink then hot therefore by the transitive property cat also implies ", return_tensors="pt")
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # Compare parameters of the first layer (as an example)
    original_params = dict(original_model.named_parameters())
    fine_tuned_params = dict(model.named_parameters())

    # Select a specific layer to compare
    print(original_params.keys())
    for layer_name in list(original_params.keys()): # Example: "embeddings.word_embeddings.weight"

        # Check if the parameters in this layer are the same
        are_parameters_different = not torch.equal(original_params[layer_name], fine_tuned_params[layer_name])
        print(f"Parameters of the layer '{layer_name}' are {'different' if are_parameters_different else 'the same'} between the models.")


if __name__ == "__main__":
    prompts_ds = get_random_nouns_ds()
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # tokenizer.pad_token = tokenizer.eos_token
    # print(tokenizer.decode(prompts_ds[23]["input_ids"], skip_special_tokens=True))
    huggingface_finetune(prompts_ds, "tunedmodels/gpt_transprop_finetune_50p_20ep", train_perc=.5, eval_perc=.1)
    #test_finetuned_model("tunedmodels/gpt2_test2")
    # train = load_pickled_data("data/train.pkl")
    # print(len(train))