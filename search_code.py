from pathlib import Path
import pickle
import time

import torch
from annoy import AnnoyIndex

from models import TextEncoder
from preprocessing_utils import VOCAB_SIZE, tokenize

model_path ="./text_encoder.ckpt"

output_file = "./code_vectors.pkl"

with open(output_file, "rb") as f:
    samples = pickle.load(f)

t = AnnoyIndex(len(samples[0]["vector"]), "angular")

for i, sample in enumerate(samples):
    t.add_item(i, sample["vector"])

t.build(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TextEncoder(lr=1e-4, vocab_size=VOCAB_SIZE)

model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu'))["state_dict"])

model.eval()
model.to(device)

# Dry run
with torch.no_grad():
    _ = model.encode(torch.tensor([[0, 0, 0]], dtype=torch.long).to(device))




def searchCode(text):
    
    search_term = text

    
    start_time_encoding = time.time()
    x1 = tokenize(search_term).ids
    x1 = torch.tensor([x1], dtype=torch.long).to(device)

    with torch.no_grad():
        search_vector = model.encode(x1).squeeze().cpu().numpy().tolist()
    end_time_encoding = time.time()

    start_time_search = time.time()
    indexes = t.get_nns_by_vector(search_vector, n=3, search_k=-1)
    end_time_search = time.time()

    results=[]
    for i, index in enumerate(indexes):
        print(f"# Search Result {i+1} -->")
        sample = samples[index]
        results.append(sample["code"])
    
    return results