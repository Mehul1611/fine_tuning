from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import torch

class ModelTrainer:
    def __init__(self, base_model_path, tokenizer, training_args, dataset):
        self.base_model_path = base_model_path
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.dataset = dataset

    def prepare_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj','fc_in', 'fc_out']
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.gradient_checkpointing_enable()

        return model

    def train_model(self, model):
        trainer = SFTTrainer(
            model=model,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['test'],
            peft_config=model.peft_config,
            max_seq_length=512,
            dataset_text_field="text",
            tokenizer=self.tokenizer,
            args=self.training_args,
            packing=False,
        )

        model.train()
        trainer.train()

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)
