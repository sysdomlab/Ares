import os
import time
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import transformers
from datasets import load_dataset, load_from_disk

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# >>>>> Model loading TODO
model_name = "facebook/opt-125m"  # "facebook/opt-1.3b"
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(model)
print(tokenizer)

# >>>>> Post-processing on the model
for param in model.parameters():
    param.requires_grad = False  # freeze the model - train adapters later

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()


# >>>>> Apply LoRA
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} "
          f"|| trainable%: {100 * trainable_params / all_param}")


config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# >>>>> Dataset
# Abirate/english_quotes
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples['quote']), batched=True, batch_size=100)
print(data["train"]["quote"][0])


## https://huggingface.co/datasets/bookcorpus  4.5GB
# data = load_dataset("bookcorpus")
# data.save_to_disk("/mnt/Datasets/bookcorpus")
# data = load_from_disk("/mnt/Datasets/bookcorpus")
# data = load_from_disk("/home/yfliu/workload_for_Athena/LLM-train/data/bookcorpus", keep_in_memory=True)
epoch = 3
batch_size = 32
steps = epoch * data["train"].num_rows // batch_size
# data["train"] = (data["train"]
#                  .to_iterable_dataset()
#                  .map(lambda samples: tokenizer(samples['text'])))
# data["train"] = (data["train"]
#                  .shuffle()
#                  .map(lambda samples: tokenizer(samples['text'])))

# >>>>> Training
trainer = transformers.Trainer(
    model=model,
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        max_steps=steps,
        # num_train_epochs=3,
        # save_steps=10000,
        # dataloader_num_workers=4,
        output_dir='outputs',
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
t1 = time.time()
trainer.train()
t2 = time.time()
print(f"\nfinished training in {t2 - t1} seconds\n")

# >>>>> Inference
batch = tokenizer("Two things are infinite: ", return_tensors='pt').to(device)
print(batch)
with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=50)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
