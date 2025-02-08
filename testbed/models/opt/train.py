import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification
from transformers import AdamW, get_scheduler
import torch
from tqdm import tqdm
from datasets import load_metric

# https://beyondguo.github.io/nlp_basis/notes/HuggingfaceNLP-8.%20%E4%BD%BF%E7%94%A8PyTorch%E6%9D%A5%E5%BE%AE%E8%B0%83.html#_1-%E6%95%B0%E6%8D%AE%E9%9B%86%E9%A2%84%E5%A4%84%E7%90%86

raw_datasets = load_dataset("glue", "mrpc")   #
checkpoint = "/home/cchen/yfliu/ares/data/squad/model/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print(tokenized_datasets['train'].column_names)

tokenized_datasets = tokenized_datasets.remove_columns(['sentence1', 'sentence2', 'idx'])
# tokenized_datasets = tokenized_datasets.rename_column('label','labels')  # 实践证明，这一行是不需要的
tokenized_datasets.set_format('torch')

print(tokenized_datasets['train'].column_names)

train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=8, collate_fn=data_collator)   #

# 查看一下train_dataloader的元素长啥样
for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})
# 可见都是长度为72，size=8的batch


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)   #

print(model(**batch))

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
lr_scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_epochs)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    for batch in tqdm(train_dataloader):
        # 要在GPU上训练，需要把数据集都移动到GPU上：
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    lr_scheduler.step()

metric = load_metric("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():  # evaluation的时候不需要算梯度
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    # 由于dataloader是每次输出一个batch，因此我们要等着把所有batch都添加进来，再进行计算
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
