import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer,
    pipeline,
)
import json
import pandas as pd

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.float16, load_in_8bit=True
)

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.do_sample = True

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    generation_config=generation_config,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    streamer=streamer,
)

with open('BioASQ-training12b/training12b_new.json') as file:
    training_list = json.load(file)['questions']

training_df = pd.json_normalize(training_list)
training_yesno = training_df[training_df.type == 'yesno'].reset_index(drop=True)
text = training_yesno.body.iloc[0]

result = llm(text)

with open('result.txt', 'w') as file:
    file.write(result)

