#!/usr/bin/env python
# coding: utf-8

# In[3]:


#pip install -Uqqq pip --progress-bar off
#pip install -qqq torch --progress-bar off
#pip install -qqq transformers --progress-bar off
#pip install -qqq accelerate --progress-bar off
#pip install -qqq bitsandbytes --progress-bar off


# In[4]:


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer,
    pipeline,
)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.float16, load_in_8bit=True
)

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.do_sample = True


# In[7]:


#get_ipython().system('pip show accelerate')


# In[3]:


streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


# In[4]:


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


# In[5]:


text = "<s>[INST] What are the pros/cons of ChatGPT vs Open Source LLMs? [/INST]"


# In[6]:


get_ipython().run_cell_magic('time', '', 'result = llm(text)')


# In[7]:


def format_prompt(prompt, system_prompt=""):
    if system_prompt.strip():
        return f"<s>[INST] {system_prompt} {prompt} [/INST]"
    return f"<s>[INST] {prompt} [/INST]"


# In[8]:


SYSTEM_PROMPT = """
You're a salesman and beet farmer know as Dwight K Schrute from the TV show The Office. Dwgight replies just as he would in the show.
You always reply as Dwight would reply. If you don't know the answer to a question, please don't share false information.
""".strip()


# In[9]:


get_ipython().run_cell_magic('time', '', 'prompt = """\nWrite an email to a new client to offer a subscription for a paper supply for 1 year.\n""".strip()\nresult = llm(format_prompt(prompt, SYSTEM_PROMPT))')


# In[10]:


get_ipython().run_cell_magic('time', '', 'prompt = """\nI have $10,000 USD for investment. How one should invest it during times of high inflation and high mortgate rates?\n""".strip()\nresult = llm(format_prompt(prompt, SYSTEM_PROMPT))')


# In[11]:


get_ipython().run_cell_magic('time', '', 'prompt = """\nWhat is the annual profit of Schrute Farms?\n""".strip()\nresult = llm(format_prompt(prompt, SYSTEM_PROMPT))')


# ## Coding

# In[12]:


get_ipython().run_cell_magic('time', '', 'prompt = """\nWrite a function in python that calculates the square of a sum of two numbers.\n""".strip()\nresponse = llm(format_prompt(prompt))')


# In[23]:


def sum_square(a, b):
    result = a + b
    return result**2


# In[24]:


sum_square(2, 3)


# In[15]:


get_ipython().run_cell_magic('time', '', 'prompt = """\nWrite a function in python that splits a list into 3 equal parts and returns a list\nwith a random element of each sublist.\n""".strip()\nresponse = llm(format_prompt(prompt))')


# In[21]:


import random


def split_list_into_3_equal_parts(lst):
    # Split the list into 3 equal parts
    parts = [lst[i : i + len(lst) // 3] for i in range(0, len(lst), len(lst) // 3)]

    # Randomly select an element from each sublist
    random_elements = [random.choice(part) for part in parts]

    # Combine the random elements into a single list
    return random_elements


# In[22]:


split_list_into_3_equal_parts([1, 2, 3, 4, 5, 6])


# ## QA over Text

# In[18]:


get_ipython().run_cell_magic('time', '', '\ntext = """\nIn this work, we develop and release Llama 2, a collection of pretrained and fine-tuned\nlarge language models (LLMs) ranging in scale from 7 billion to 70 billion parameters.\nOur fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Our\nmodels outperform open-source chat models on most benchmarks we tested, and based on\nour human evaluations for helpfulness and safety, may be a suitable substitute for closedsource models. We provide a detailed description of our approach to fine-tuning and safety\nimprovements of Llama 2-Chat in order to enable the community to build on our work and\ncontribute to the responsible development of LLMs.\n"""\n\nprompt = f"""\nUse the text to describe the benefits of Llama 2:\n{text}\n""".strip()\n\nresponse = llm(format_prompt(prompt))')


# ## Data Extraction

# In[19]:


get_ipython().run_cell_magic('time', '', 'table = """\n|Model|Size|Code|Commonsense Reasoning|World Knowledge|Reading Comprehension|Math|MMLU|BBH|AGI Eval|\n|---|---|---|---|---|---|---|---|---|---|\n|Llama 1|7B|14.1|60.8|46.2|58.5|6.95|35.1|30.3|23.9|\n|Llama 1|13B|18.9|66.1|52.6|62.3|10.9|46.9|37.0|33.9|\n|Llama 1|33B|26.0|70.0|58.4|67.6|21.4|57.8|39.8|41.7|\n|Llama 1|65B|30.7|70.7|60.5|68.6|30.8|63.4|43.5|47.6|\n|Llama 2|7B|16.8|63.9|48.9|61.3|14.6|45.3|32.6|29.3|\n|Llama 2|13B|24.5|66.9|55.4|65.8|28.7|54.8|39.4|39.1|\n|Llama 2|70B|**37.5**|**71.9**|**63.6**|**69.4**|**35.2**|**68.9**|**51.2**|**54.2**|\n"""\n\nprompt = f"""\nUse the data from the markdown table:\n\n```\n{table}\n```\n\nto answer the question:\nExtract the Reading Comprehension score for Llama 2 7B\n"""\n\nresponse = llm(format_prompt(prompt))')


# In[20]:


get_ipython().run_cell_magic('time', '', 'table = """\n|Model|Size|Code|Commonsense Reasoning|World Knowledge|Reading Comprehension|Math|MMLU|BBH|AGI Eval|\n|---|---|---|---|---|---|---|---|---|---|\n|Llama 1|7B|14.1|60.8|46.2|58.5|6.95|35.1|30.3|23.9|\n|Llama 1|13B|18.9|66.1|52.6|62.3|10.9|46.9|37.0|33.9|\n|Llama 1|33B|26.0|70.0|58.4|67.6|21.4|57.8|39.8|41.7|\n|Llama 1|65B|30.7|70.7|60.5|68.6|30.8|63.4|43.5|47.6|\n|Llama 2|7B|16.8|63.9|48.9|61.3|14.6|45.3|32.6|29.3|\n|Llama 2|13B|24.5|66.9|55.4|65.8|28.7|54.8|39.4|39.1|\n|Llama 2|70B|**37.5**|**71.9**|**63.6**|**69.4**|**35.2**|**68.9**|**51.2**|**54.2**|\n"""\n\nprompt = f"""\nUse the data from the markdown table:\n\n```\n{table}\n```\n\nto answer the question:\nCalculate how much better (% increase) is Llama 2 7B vs Llama 1 7B on Reading Comprehension?\n"""\n\nresponse = llm(format_prompt(prompt))')


# ## References
#
# - [Mistral Home Page](https://mistral.ai/)
# - [Mistral 7B Paper](https://arxiv.org/pdf/2310.06825.pdf)
# - [Mistral-7B-Instruct-v0.1 on HuggingFace](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
# - [Mistral System Prompt](https://docs.mistral.ai/usage/guardrailing/#appendix)
