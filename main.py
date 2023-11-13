import sys
import pandas as pd 
import sqlite3 

from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chat_models import ChatOpenAI 
from langchain.prompts import PromptTemplate 

import torch
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline 

df1 = pd.read_csv('DS_NLP_search_data/brand_category.csv')
df2 = pd.read_csv('DS_NLP_search_data/categories.csv')
df3 = pd.read_csv('DS_NLP_search_data/offer_retailer.csv')

# my_dfs = {
#     "brand": df1,
#     "categories": df2,
#     "offers": df3,
# }
# with sqlite3.connect("DS_NLP_search_data/my_db.sqlite") as my_db:
#     for table_name, df in my_dfs.items():
#         df.to_sql(table_name, my_db, if_exists="replace")


model_name = 'TheBloke/Llama-2-13B-chat-GPTQ'
# model_name = "meta-llama/Llama-2-7b-chat-hf"

cache_dir = 'cache_dir'

print('loading tokenizer....', end='')
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
print('done!')

print('loading model....', end='')
model = AutoModelForCausalLM.from_pretrained(
    model_name, cache_dir=cache_dir, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True,
)
print('done!')

print('loading gen_config....',end='')
generation_config = GenerationConfig.from_pretrained(model_name)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15
print('done!')

text_pipeline = pipeline(
    'text-generation',
    model = model,
    tokenizer = tokenizer,
    return_full_text=True,
    generation_config=generation_config,
)

llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={'temperature' : 0.})

result = llm(
    'Explain Machine Learning in 2 sentences'
)

print(result)

db = SQLDatabase.from_uri('sqlite:///my_db.sqlite')
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
query = "Get me offers on AMAZON"

template = """
You are an offer finding assistant. You will check the tables and retrieve the most relevant offers \
given the query. Given an input query, first create a syntactically correct sqlite query to run, \
then look at the results of the query and return the answer. There are 3 tables in the database. \
There descriptions are as follows:
1. brand: contain the name of the brand and product categories that brand sell.
2. categories: contain different types of product categories
3. offers: contain brands and corresponding offers on different product categories. 
Do not make up offers. Return the most relevant offers. 
Query:{query}"""

prompt = PromptTemplate(
    input_variables=['query'],
    template=template
)

db_chain.run(prompt.format(query= query))