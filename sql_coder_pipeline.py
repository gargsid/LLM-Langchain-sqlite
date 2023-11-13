import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse

model_name = "defog/sqlcoder"
cache_dir='cache_dir'
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    # load_in_8bit=True,
    # load_in_4bit=True,
    device_map="auto",
    use_cache=True,
)

question = "What products has the biggest fall in sales in 2022 compared to 2021? Give me the product name, the sales amount in both years, and the difference."

prompt = """### Instructions:
Your task is convert a question into a SQL query, given a Postgres database schema.
Adhere to these rules:
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
- When creating a ratio, always cast the numerator as float

### Input:
Generate a SQL query that answers the questioq `{question}`.
This query will run on a database whose schema is represented in this string:
CREATE TABLE products (
  product_id INTEGER PRIMARY KEY, -- Unique ID for each product
  name VARCHAR(50), -- Name of the product
  price DECIMAL(10,2), -- Price of each unit of the product
  quantity INTEGER  -- Current quantity in stock
);

CREATE TABLE customers (
   customer_id INTEGER PRIMARY KEY, -- Unique ID for each customer
   name VARCHAR(50), -- Name of the customer
   address VARCHAR(100) -- Mailing address of the customer
);

CREATE TABLE salespeople (
  salesperson_id INTEGER PRIMARY KEY, -- Unique ID for each salesperson
  name VARCHAR(50), -- Name of the salesperson
  region VARCHAR(50) -- Geographic sales region
);

CREATE TABLE sales (
  sale_id INTEGER PRIMARY KEY, -- Unique ID for each sale
  product_id INTEGER, -- ID of product sold
  customer_id INTEGER,  -- ID of customer who made purchase
  salesperson_id INTEGER, -- ID of salesperson who made the sale
  sale_date DATE, -- Date the sale occurred
  quantity INTEGER -- Quantity of product sold
);

CREATE TABLE product_suppliers (
  supplier_id INTEGER PRIMARY KEY, -- Unique ID for each supplier
  product_id INTEGER, -- Product ID supplied
  supply_price DECIMAL(10,2) -- Unit price charged by supplier
);

-- sales.product_id can be joined with products.product_id
-- sales.customer_id can be joined with customers.customer_id
-- sales.salesperson_id can be joined with salespeople.salesperson_id
-- product_suppliers.product_id can be joined with products.product_id

### Response:
Based on your instructions, here is the SQL query I have generated to answer the question `{question}`:
```sql
""".format(question=question)

eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
generated_ids = model.generate(
    **inputs,
    num_return_sequences=1,
    eos_token_id=eos_token_id,
    pad_token_id=eos_token_id,
    max_new_tokens=400,
    do_sample=False,
    num_beams=5 
)

outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

torch.cuda.empty_cache()

print(outputs[0])

print('-'*20)

print(outputs[0].split("```sql")[-1].split("```")[0].split(";")[0].strip() + ";")