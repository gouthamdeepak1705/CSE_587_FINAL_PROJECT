# 1) Set your token
import os

# 2) Load all
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, use_fast=True, token=os.environ["HUGGINGFACE_TOKEN"]
)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    token=os.environ["HUGGINGFACE_TOKEN"]
)

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16, lora_alpha=32, lora_dropout=0.05
)
model = get_peft_model(model, lora_cfg)
model.resize_token_embeddings(len(tokenizer))

print(" Loaded 8-bit + LoRA Llama-2 successfully!")
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
import os
import torch
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    ProgressCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
import shutil

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
HF_TOKEN = os.environ["HUGGINGFACE_TOKEN"]

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
EXCEL_PATH = "/kaggle/input/abcdefg/summaries_split.xlsx"
MAX_LEN = 256
OFFLOAD_DIR = "offload"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

current_device = torch.cuda.current_device()

df = pd.read_excel(EXCEL_PATH)

def build_pairs(df):
    rec = []
    for _, row in df.iterrows():
        title, summ = row["Title"], str(row["Summary"]).strip()
        typ  = row.get("Type")     if pd.notna(row.get("Type"))     else None
        dset = row.get("Datasets") if pd.notna(row.get("Datasets")) else None

        rec.append({"instruction":"How do I solve LLM hallucination?","input":"","output":summ})
        if typ:
            rec.append({"instruction":f"How would you solve LLM hallucination using a {typ} approach?",
                        "input":"","output":summ})
        if dset:
            rec.append({"instruction":f"How would you solve LLM hallucination on {dset}?","input":"","output":summ})
        rec.append({"instruction":f"Tell me about the {title} method.","input":"","output":summ})
    return pd.DataFrame(rec)

qa_df = build_pairs(df)
ds    = Dataset.from_pandas(qa_df).train_test_split(test_size=0.1, seed=42)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    token=HF_TOKEN,
    trust_remote_code=True
)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

def make_prompt(instr, inp, out):
    if inp:
        return (
            "### Instruction:\n" + instr +
            "\n\nInput:\n"        + inp +
            "\n\n### Response:\n" + out
        )
    return "### Instruction:\n" + instr + "\n\n### Response:\n" + out

def tokenize_batch(batch):
    texts = [
        make_prompt(i, inp, out)
        for i, inp, out in zip(batch["instruction"], batch["input"], batch["output"])
    ]
    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized = ds.map(
    tokenize_batch,
    batched=True,
    remove_columns=ds["train"].column_names,
    desc="Tokenising"
)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_cfg,
    device_map={"": current_device},
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,
    trust_remote_code=True,
    token=HF_TOKEN
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    inference_mode=False
)
model = get_peft_model(model, lora_cfg)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="llama2-ft-qas",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    fp16=True,
    optim="paged_adamw_8bit",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    report_to=[],
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

print("Starting training")
trainer.train()

print("Saving model and tokenizer to local folder 'llama2-ft-qas'")
trainer.save_model("llama2-ft-qas")
tokenizer.save_pretrained("llama2-ft-qas")

print("Zipping adapter folder")
shutil.make_archive("llama2-ft-qas", "zip", "llama2-ft-qas")
print("Done. Download '/kaggle/working/llama2-ft-qas.zip' for your LoRA adapter.")

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel
import shutil

HF_TOKEN    = os.environ["HUGGINGFACE_TOKEN"]
MODEL_NAME  = "meta-llama/Llama-2-7b-chat-hf"
ADAPTER_DIR = "llama2-ft-qas"              # your first adapter folder
OFFLOAD_DIR = "offload"
NEW_QA_PATH = "/kaggle/input/qa-final/QA_final.xlsx"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

device_id = torch.cuda.current_device()

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_cfg,
    device_map={ "": device_id },
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,
    trust_remote_code=True,
    use_auth_token=HF_TOKEN,
)
base.gradient_checkpointing_enable()
base.enable_input_require_grads()
base.config.use_cache = False

model = PeftModel.from_pretrained(
    base,
    ADAPTER_DIR,
    device_map={ "": device_id },
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    trust_remote_code=True,
    use_auth_token=HF_TOKEN,
)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

df2 = pd.read_excel(NEW_QA_PATH)
records = [
    {"instruction": r["Question"], "input": "", "output": str(r["Answer"]).strip()}
    for _, r in df2.iterrows()
]
qa2 = pd.DataFrame(records)
ds2 = Dataset.from_pandas(qa2).train_test_split(test_size=0.1, seed=42)

MAX_LEN = 256
def make_prompt(i, inp, out):
    if inp:
        return f"### Instruction:\n{i}\n\nInput:\n{inp}\n\n### Response:\n{out}"
    return f"### Instruction:\n{i}\n\n### Response:\n{out}"

def tokenize_batch(batch):
    texts = [make_prompt(i, inp, out)
             for i, inp, out in zip(batch["instruction"], batch["input"], batch["output"])]
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN)
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized2 = ds2.map(
    tokenize_batch,
    batched=True,
    remove_columns=ds2["train"].column_names,
    desc="Tokenising in_depth QA"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir="llama2-ft-qas-cont",
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    fp16=True,
    optim="paged_adamw_8bit",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    report_to=[],
    push_to_hub=False,
)

trainer2 = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized2["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Starting second fine-tuning on in-depth QA dataset…")
trainer2.train()
print("Second fine-tuning complete.")

trainer2.save_model("llama2-ft-qas-cont")
tokenizer.save_pretrained("llama2-ft-qas-cont")
shutil.make_archive("llama2-ft-qas-cont", "zip", "llama2-ft-qas-cont")
print("Done. Download '/kaggle/working/llama2-ft-qas-cont.zip'")

import os
import gc
import torch
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

HF_TOKEN   = os.environ["HUGGINGFACE_TOKEN"]
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
QA_PATH    = "/kaggle/input/qa-final/QA_final.xlsx"
OUTPUT_DIR = "llama2-ft-qa_only"

gc.collect()
torch.cuda.empty_cache()

df = pd.read_excel(QA_PATH)[["Question","Answer"]].dropna().reset_index(drop=True)
records = []
for _, row in df.iterrows():
    records.append({
        "instruction": row["Question"].strip(),
        "input": "",
        "output": str(row["Answer"]).strip(),
    })
qa_df = pd.DataFrame(records)
ds = Dataset.from_pandas(qa_df).train_test_split(test_size=0.1, seed=42)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, use_fast=True, token=HF_TOKEN, trust_remote_code=True
)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

MAX_LEN = 256
def make_prompt(inst, inp, out):
    return f"### Instruction:\n{inst}\n\n### Response:\n{out}"

def tokenize_fn(batch):
    texts = [make_prompt(i, inp, out)
             for i, inp, out in zip(batch["instruction"], batch["input"], batch["output"])]
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN)
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized = ds.map(
    tokenize_fn,
    batched=True,
    remove_columns=ds["train"].column_names,
    desc="Tokenizing QA"
)

device = torch.cuda.current_device()
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_cfg,
    device_map={"": device},
    trust_remote_code=True,
    token=HF_TOKEN,
)
base_model.gradient_checkpointing_enable()
base_model.enable_input_require_grads()
base_model.config.use_cache = False

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    inference_mode=False,
)
model = get_peft_model(base_model, lora_cfg).to(device)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    fp16=True,
    optim="paged_adamw_8bit",
    logging_steps=50,
    save_steps=200,
    save_total_limit=3,
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f" Fine-tuning complete! Adapter & tokenizer saved to {OUTPUT_DIR}/")