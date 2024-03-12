import os
import requests
import tiktoken
import numpy as np
from datasets import load_dataset

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    with open(input_file_path, 'a') as output_file:
        # Iterate through each text file in the data folder
        for filename in os.listdir('data'):
            # Check if the file is a text file
            if filename.endswith('.txt'):
                # Open the text file
                with open(os.path.join('data', filename), 'r') as input_file:
                    # Read the content of the text file
                    content = input_file.read()
                    # Write the content to the output file
                    output_file.write(content)



with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

data = "" # currently we will only be using huggingface datasets, but later we can also add our own files/documents
# load in huggingface datasets
electronics_dataset = load_dataset("ksabeh/electronics-dataset")
fabner_dataset = load_dataset("DFKI-SLT/fabner")

for i in range(len(electronics_dataset['train'])):
    data += electronics_dataset['train'][i]['text']

for i in range(len(fabner_dataset['train'])):
    tokens = fabner_dataset['train'][i]['tokens']

    data += ' '.join(tokens)

n = len(data)
print(n)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 1,860,788 tokens
# val.bin has 123,117 tokens
