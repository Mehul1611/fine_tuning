from huggingface_hub import login
import wandb
from kaggle_secrets import UserSecretsClient

class Authenticator:
    def __init__(self):
        self.user_secrets = UserSecretsClient()
        
    def authenticate(self):
        hf_token = self.user_secrets.get_secret("HUGGINGFACE_TOKEN")
        wb_token = self.user_secrets.get_secret("wandb")
        
        login(token=hf_token)
        wandb.login(key=wb_token)
