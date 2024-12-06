import torch

class ChatModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def generate_response(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

        outputs = self.model.generate(**inputs, max_length=150, num_return_sequences=1)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return text
