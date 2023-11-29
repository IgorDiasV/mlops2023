import pytest
import wandb
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = os.environ.get('PROJECT_NAME')
run = wandb.init(project=PROJECT_NAME, job_type="data_checks")

@pytest.fixture(scope="session")
def data():
    local_path = run.use_artifact("clean_data:latest").download()
    return local_path

def test_file_existence(data):
    assert os.path.exists(os.path.join(data, 'clean_data.csv'))

def test_non_empty_file(data):
  assert os.path.getsize(os.path.join(data, 'clean_data.csv')) > 0