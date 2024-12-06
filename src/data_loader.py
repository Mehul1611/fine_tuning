import pandas as pd
from datasets import Dataset, load_dataset

class DataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
    def load_data(self):
        # Load dataset from JSONL file
        dataset = load_dataset("json", data_files={"train": self.dataset_path})
        original_df = pd.read_json(self.dataset_path, lines=True)
        return Dataset.from_pandas(original_df)
    
    def format_chat_template(self, row, tokenizer):
        messages = row['messages']
        row_json = [
            {"role": "system", "content": messages[0].get('content', '')},
            {"role": "user", "content": messages[1].get('content', '')},
            {"role": "assistant", "content": messages[2].get('content', '')}
        ]
        row["text"] = tokenizer.apply_chat_template(
            row_json, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        return row

    def preprocess_data(self, dataset, tokenizer):
        return dataset.map(lambda row: self.format_chat_template(row, tokenizer))
