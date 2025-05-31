import json
import os
from tqdm import tqdm
from collections import defaultdict

all_data = []

old_date = "241101"
new_date = "241201"

with open(f"../data/diff/diff_{old_date}_{new_date}_final.jsonl") as f:
    for line in f:
        all_data.append(json.loads(line))

id2datalist = defaultdict(list)

for data in all_data:
    id2datalist[int(data["id"]) // 1000000].append(data)

old_dir_path = f"../data/enwiki/enwiki-20{old_date}_cleaned"
new_dir_path = f"../data/enwiki/enwiki-20{new_date}_cleaned"

for id, datalist in tqdm(id2datalist.items()):
    file_old = os.path.join(old_dir_path, f"{id}.jsonl")
    with open(file_old, "r") as f:
        id2oldpassage = {}
        for line in f:
            data = json.loads(line)
            id2oldpassage[data["id"]] = data["text"]
    for data_ in datalist:
        data_["text_old"] = id2oldpassage[data_["id"]]
    file_new = os.path.join(new_dir_path, f"{id}.jsonl")
    with open(file_new, "r") as f:
        id2newpassage = {}
        for line in f:
            data = json.loads(line)
            id2newpassage[data["id"]] = data["text"]
    for data_ in datalist:
        data_["text_new"] = id2newpassage[data_["id"]]

with open(f"../data/diff/diff_{old_date}_{new_date}_text.jsonl", "w") as f:
    for data_ in all_data:
        f.write(json.dumps(data_) + "\n")