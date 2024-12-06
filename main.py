from src.config import HUGGINGFACE_TOKEN, WANDB_SECRET_KEY, BASE_MODEL_PATH, NEW_MODEL_PATH, DATASET_PATH, TRAINING_ARGS
from src.authentication import Authenticator
from src.data_loader import DataLoader
from src.model_trainer import ModelTrainer
from src.chat_model import ChatModel
from transformers import AutoTokenizer

def main():
    # Step 1: Authentication
    authenticator = Authenticator()
    authenticator.authenticate()

    # Step 2: Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    data_loader = DataLoader(DATASET_PATH)
    dataset = data_loader.load_data()
    dataset = data_loader.preprocess_data(dataset, tokenizer)

    # Step 3: Prepare the model
    model_trainer = ModelTrainer(BASE_MODEL_PATH, tokenizer, TRAINING_ARGS, dataset)
    model = model_trainer.prepare_model()

    # Step 4: Train the model
    model_trainer.train_model(model)

    # Step 5: Save the model
    model_trainer.save_model(model, '/kaggle/working/finalized_model.pth')

    # Step 6: Generate response (Example)
    chat_model = ChatModel(model, tokenizer)
    messages = [
        {"role": "system", "content": "You are a helpful assistant who answers questions about book titles and authors."},
        {"role": "user", "content": "Who is the author of the book 'The Middle Stories'?"}
    ]
    response = chat_model.generate_response(messages)
    print(response)

if __name__ == "__main__":
    main()
