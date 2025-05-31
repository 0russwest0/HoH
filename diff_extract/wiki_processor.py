import os
import json
from tqdm import tqdm
from collections import defaultdict
import diff_match_patch as dmp_module
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
from nltk.tokenize import sent_tokenize
from pathlib import Path
import yaml

spacy.prefer_gpu()

class WikiDiffProcessor:
    def __init__(self, config=None):
        """
        Initialize WikiDiffProcessor
        Args:
            config: Configuration dictionary from YAML file containing model, data and processing settings
        """
        if config is None:
            # 如果没有提供配置，加载默认配置
            config_path = Path(__file__).parent / "config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        
        self.config = config
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config["model"]["cuda_devices"]
        self.dmp = dmp_module.diff_match_patch()
        
        # Setup sentence tokenizer
        self.sentence_tokenizer = self.config["processing"]["sentence_tokenizer"].lower()
        if self.sentence_tokenizer == "spacy":
            self.nlp = spacy.load("en_core_web_sm")
        elif self.sentence_tokenizer == "nltk":
            self.split_sentences = lambda text: sent_tokenize(text)
        elif self.sentence_tokenizer == "simple":
            self.split_pattern = re.compile(r'([.!?]+[\s\n]+)')
        else:
            raise ValueError("sentence_tokenizer in config must be one of: 'spacy', 'nltk', 'simple'")
        
        # Load special diffs
        with open(self.config["processing"]["special_diffs_path"], "r") as f:
            self.special_diffs = json.load(f)
    
    def split_sentences(self, text):
        """
        Split text into sentences using the selected tokenizer
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        if self.sentence_tokenizer == "spacy":
            return [sent.text.strip() for sent in self.nlp(text).sents]
        elif self.sentence_tokenizer == "nltk":
            return sent_tokenize(text)
        else:  # simple
            # Split on sentence endings and keep the delimiters
            sentences = self.split_pattern.split(text)
            # Combine each sentence with its delimiter and filter empty strings
            result = []
            for i in range(0, len(sentences)-1, 2):
                if sentences[i].strip():
                    result.append((sentences[i] + sentences[i+1]).strip())
            # Add the last sentence if it exists and isn't empty
            if len(sentences) % 2 == 1 and sentences[-1].strip():
                result.append(sentences[-1].strip())
            return result

    def process_raw_data(self, raw_data_path, cleaned_data_path):
        """Process raw Wikipedia data, clean and save grouped by ID."""
        dir_paths = os.listdir(raw_data_path)
        os.makedirs(cleaned_data_path, exist_ok=True)

        # Collect all file paths
        file_paths = []
        for dir_path in dir_paths:
            paths = os.listdir(os.path.join(raw_data_path, dir_path))
            for path in paths:
                file_paths.append(os.path.join(raw_data_path, dir_path, path))
        
        data_cleaned = defaultdict(list)

        for file_path in tqdm(file_paths):
            with open(file_path, 'r') as f:
                for line in f:
                    json_data = json.loads(line)
                    id = json_data["id"]
                    text = json_data["text"]
                    if len(text) < self.config["processing"]["min_text_length"]:
                        continue
                    id_ = int(id) // 1000000
                    data_cleaned[id_].append(json_data)

        for key in data_cleaned.keys():
            with open(os.path.join(cleaned_data_path, f"{key}.jsonl"), 'w') as f:
                for data in data_cleaned[key]:
                    f.write(json.dumps(data) + '\n')

    def _split_diff_to_pairs(self, diff):
        """Split diff text into line-level pairs of changes."""
        original, modified = diff
        original_sentences = original.split("\n")
        modified_sentences = modified.split("\n")
        
        if original.count("\n") <= 1 and modified.count("\n") <= 1:
            return [[original.strip(), modified.strip()]]
            
        if original[-1] != "\n": original += "\n"
        if modified[-1] != "\n": modified += "\n"
        
        raw_diffs = self.dmp.diff_main(original, modified)
        o_i = m_i = 0
        sentence_diffs = []
        
        for op, text in raw_diffs:
            num = text.count("\n")
            if num > 0:
                if op == 0:
                    sentence_diffs.append([original_sentences[o_i], modified_sentences[m_i]])
                    o_i += num
                    m_i += num
                elif op == -1:
                    o_i += num
                elif op == 1:
                    m_i += num
                    
        return sentence_diffs

    def extract_diffs(self, old_path, new_path, diff_path):
        """Compare Wikipedia data from two time points and generate diff file."""
        diff_data = []
        files = os.listdir(old_path)
        
        for file in tqdm(files):
            old_data = {}
            new_data = {}
            
            # Load old and new data
            with open(os.path.join(old_path, file), "r") as f:
                for line in f:
                    data = json.loads(line)
                    old_data[data["id"]] = data
            
            with open(os.path.join(new_path, file), "r") as f:
                for line in f:
                    data = json.loads(line)
                    new_data[data["id"]] = data
            
            # Compare and extract diffs
            for key in old_data:
                if key in new_data and old_data[key]['text'] != new_data[key]['text']:
                    data = {
                        "id": key,
                        "title": old_data[key]['title'],
                        "old_text": old_data[key]['text'],
                        "new_text": new_data[key]['text']
                    }
                    diff_data.append(data)

        # Process and save diffs
        with open(diff_path, "w") as f:   
            for data in tqdm(diff_data):
                old_text = data['old_text']
                new_text = data['new_text']
                raw_diffs = self.dmp.diff_main(old_text.lower(), new_text.lower())
                raw_diffs_type = [diff[0] for diff in raw_diffs]
                
                if -1 in raw_diffs_type and 1 in raw_diffs_type:
                    old_sentences = '\n'.join(self.split_sentences(old_text))
                    new_sentences = '\n'.join(self.split_sentences(new_text))
                    a = self.dmp.diff_linesToChars(old_sentences, new_sentences)
                    diffs = self.dmp.diff_main(a[0], a[1], False)
                    self.dmp.diff_charsToLines(diffs, a[2])
                    
                    pairs = []
                    for i in range(1, len(diffs)):
                        if diffs[i - 1][0] == -1 and diffs[i][0] == 1:
                            if diffs[i - 1][1].lower() == diffs[i][1].lower():
                                continue
                            pairs.extend(self._split_diff_to_pairs([diffs[i - 1][1], diffs[i][1]]))
                    
                    def verify_diff_pair(old_text, new_text, pair):
                        old_part, new_part = pair
                        return old_part in old_text and new_part in new_text

                    if pairs:
                        verified_pairs = [
                            pair for pair in pairs 
                            if verify_diff_pair(data['old_text'], data['new_text'], pair)
                        ]
                        if verified_pairs:
                            new_data = {
                                "id": data["id"],
                                "title": data["title"],
                                "diffs": verified_pairs
                            }
                            f.write(json.dumps(new_data) + '\n')

    def _diff_filter(self, diffs):
        """Filter out unimportant diffs."""
        types = [diff[0] for diff in diffs]
        if -1 not in types or 1 not in types:
            return True

        contents = [diff[1] for diff in diffs if diff[0] != 0]
        
        for content in contents:
            if content.lower() in self.config["processing"]["word_list"]:
                continue
            elif re.match(r'^\d+$', content):
                continue
            elif len(content) <= 2:
                return True
        
        pairs = []
        for i in range(1, len(diffs)):
            if diffs[i - 1][0] == -1 and diffs[i][0] == 1:
                pair = [diffs[i - 1][1], diffs[i][1]]
                if pair in self.special_diffs:
                    continue
                if pair[0].lower() in self.config["processing"]["skip_list"] or pair[1].lower() in self.config["processing"]["skip_list"]:
                    continue
                pairs.append(pair)
                
        return len(pairs) == 0

    def filter_diffs(self, input_path, output_path):
        """Further filter diff data, remove formulas and short sentences."""
        filtered_data = []
        
        with open(input_path, "r") as f:
            for line in tqdm(f):
                data = json.loads(line)
                diffs = data["diffs"]
                filtered_diffs = []
                
                for pair in diffs:
                    original, modified = pair
                    if "_" in original or "_" in modified:
                        continue
                    if len(original.split()) <= 5 or len(modified.split()) <= 5:
                        continue
                        
                    raw_diffs = self.dmp.diff_main(original, modified)
                    self.dmp.diff_cleanupSemantic(raw_diffs)
                    
                    if not self._diff_filter(raw_diffs):
                        filtered_diffs.append(pair)
                        
                if filtered_diffs:
                    data["diffs"] = filtered_diffs
                    filtered_data.append(data)

        with open(output_path, "w") as f:
            for data in filtered_data:
                f.write(json.dumps(data) + "\n")

    def model_filter(self, input_path, output_path, model_path):
        """Filter diffs using a pre-trained model."""
        all_data = []
        with open(input_path, "r") as f:
            for line in f:
                all_data.append(json.loads(line))

        for data in all_data:
            data["diffs"] = list(set([tuple(diff) for diff in data["diffs"]]))

        all_diffs = []
        for data in all_data:
            all_diffs.extend(data["diffs"])

        del_format = '~~{delete}~~'
        ins_format = '<u>{insert}</u>'

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        all_diffs_ = []
        for diff in tqdm(all_diffs):
            diffs = self.dmp.diff_main(diff[0], diff[1])
            chars1, chars2, segmentArray = self._diff_customSegmentsToChars(diff[0], diff[1], tokenizer.tokenize)
            diffs = self.dmp.diff_main(chars1, chars2)
            self._diff_charsToSegments(diffs, segmentArray, join_function=tokenizer.convert_tokens_to_string)
            
            diff_ = ""
            for op, data in diffs:
                if len(data) == 0:
                    continue
                if op == 0:
                    diff_ += data
                elif op == -1:
                    if data[0] == " ":
                        diff_ += " " + del_format.format(delete=data[1:])
                    else:
                        diff_ += del_format.format(delete=data)
                elif op == 1:
                    if data[0] == " ":
                        diff_ += " " + ins_format.format(insert=data[1:])
                    else:
                        diff_ += ins_format.format(insert=data)
            diff_ = diff_.replace("</u> <u>", " ")
            all_diffs_.append(diff_)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=2, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        model.config.pad_token_id = model.config.eos_token_id

        all_preds = []
        batch_size = self.config["model"]["batch_size"]
        for i in tqdm(range(0, len(all_diffs_), batch_size)):
            batch = all_diffs_[i:i+batch_size]
            batch = tokenizer(batch, padding=True, return_tensors="pt")
            preds = model(**batch).logits
            preds = torch.argmax(preds, dim=1).tolist()
            all_preds.extend(preds)

        i = 0
        for data in all_data:
            filtered_diffs = []
            for diff in data["diffs"]:
                if all_preds[i] == 1:
                    filtered_diffs.append(diff)
                i += 1
            data["diffs"] = filtered_diffs

        with open(output_path, "w") as f:
            for data in all_data:
                if data["diffs"]:
                    f.write(json.dumps(data) + "\n")

    def _diff_customSegmentsToChars(self, text1, text2, split_function):
        """Split two texts using a custom segmentation function."""
        segmentArray = [""]
        segmentHash = {}
        maxSegments = 1114111

        def munge(text):
            segments = split_function(text)
            chars = []
            for segment in segments:
                if segment in segmentHash:
                    chars.append(chr(segmentHash[segment]))
                else:
                    if len(segmentArray) == maxSegments:
                        segment = text[len("".join(segments[:len(chars)])):]
                        segmentArray.append(segment)
                        segmentHash[segment] = len(segmentArray) - 1
                        chars.append(chr(len(segmentArray) - 1))
                        break
                    segmentArray.append(segment)
                    segmentHash[segment] = len(segmentArray) - 1
                    chars.append(chr(len(segmentArray) - 1))
            return "".join(chars)

        chars1 = munge(text1)
        chars2 = munge(text2)
        return (chars1, chars2, segmentArray)

    def _diff_charsToSegments(self, diffs, segmentArray, join_function=None):
        """Rehydrate the text in a diff from a string of segment hashes to real segments."""
        for i in range(len(diffs)):
            text = []
            for char in diffs[i][1]:
                text.append(segmentArray[ord(char)])
            if join_function:
                diffs[i] = (diffs[i][0], join_function(text))
            else:
                diffs[i] = (diffs[i][0], "".join(text)) 

    def add_passage(self, input_path, output_path, old_path, new_path):
        """Add original and new passages to diff data.
        
        Args:
            input_path: Path to the input diff file
            output_path: Path to save the output diff file with passages
            old_path: Directory containing old wiki data
            new_path: Directory containing new wiki data
        """
        all_data = []
        with open(input_path) as f:
            for line in f:
                all_data.append(json.loads(line))

        # Group data by file ID
        id2datalist = defaultdict(list)
        for data in all_data:
            id2datalist[int(data["id"]) // 1000000].append(data)

        # Process each group
        for id, datalist in tqdm(id2datalist.items()):
            # Load old passages
            file_old = os.path.join(old_path, f"{id}.jsonl")
            with open(file_old, "r") as f:
                id2oldpassage = {}
                for line in f:
                    data = json.loads(line)
                    id2oldpassage[data["id"]] = data["text"]
                
            # Add old passages
            for data_ in datalist:
                data_["text_old"] = id2oldpassage[data_["id"]]

            # Load new passages
            file_new = os.path.join(new_path, f"{id}.jsonl")
            with open(file_new, "r") as f:
                id2newpassage = {}
                for line in f:
                    data = json.loads(line)
                    id2newpassage[data["id"]] = data["text"]
                
            # Add new passages
            for data_ in datalist:
                data_["text_new"] = id2newpassage[data_["id"]]

        # Save results
        with open(output_path, "w") as f:
            for data_ in all_data:
                f.write(json.dumps(data_) + "\n") 