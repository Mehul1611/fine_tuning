HUGGINGFACE_TOKEN = "your_huggingface_token"
WANDB_SECRET_KEY = "your_wandb_key"

BASE_MODEL_PATH = "/kaggle/input/llama-3.2/transformers/3b-instruct/1"
NEW_MODEL_PATH = "fine-tuned-llama-8b-chat"

DATASET_PATH = "/kaggle/input/book-ft/books_chat.jsonl"

TRAINING_ARGS = {
    "output_dir": NEW_MODEL_PATH,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "optim": "paged_adamw_32bit",
    "num_train_epochs": 5,
    "evaluation_strategy": "steps",
    "eval_steps": 0.2,
    "logging_steps": 1,
    "warmup_steps": 100,
    "logging_strategy": "steps",
    "learning_rate": 2e-4,
    "fp16": True,
    "bf16": False,
    "group_by_length": True,
    "report_to": ["wandb"],
    "lr_scheduler_type": "linear",
}
