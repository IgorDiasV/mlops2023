import os
import wandb
from dotenv import load_dotenv
import subprocess

load_dotenv()

chave_api = os.environ.get('KEY')
wandb.login(key=chave_api)

subprocess.run("pytest . -vv", shell=True)

wandb.finish()