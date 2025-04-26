api_token="Please Add you hugging face api token"
from huggingface_hub import login
login(api_token)



from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

import torch, os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login


BASE = "meta-llama/Llama-2-7b-chat-hf"
LORA = Path("/kaggle/input/llama2research/llama2-ft-qas-cont/checkpoint-144")

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    quantization_config=bnb_cfg
)

model = PeftModel.from_pretrained(
    base_model,
    LORA,
    torch_dtype=torch.float16
).eval()

prompt   = "Summarize why OMOP CDM matters for healthcare AI."
msg      = f"<s>[INST] {prompt} [/INST]"
inputs   = tokenizer(msg, return_tensors="pt").to(model.device)
import datetime
start_time = datetime.datetime.now()
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )[0]
print(datetime.datetime.now()-start_time)
print(tokenizer.decode(out[inputs.input_ids.shape[-1]:], skip_special_tokens=True))

import pandas as pd
import torch
from pathlib import Path

input_excel = Path('/kaggle/input/test-set/hallucination_test_set.xlsx')
df = pd.read_excel(input_excel)

questions = df.iloc[:, 0].astype(str).tolist()

answers = []
count=0
for q in questions:
    count+=1
    print(count)
    print(q)
    prompt = f"<s>[INST] {q.strip()} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )[0]
    answer = tokenizer.decode(
        out_ids[inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()
    answers.append(answer)
    print(answer)

output_df = pd.DataFrame({
    'question': questions,
    'answer': answers
})
output_csv = Path('/kaggle/working/QA_with_answers.csv')
output_df.to_csv(output_csv, index=False)

print(f"Saved {len(questions)} question-answer pairs to {output_csv}")

