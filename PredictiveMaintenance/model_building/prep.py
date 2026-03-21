
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Configuration
DATASET_REPO = "SantoshS23/PredMaintDataSet"

# Get Hugging Face token from environment variables
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set. Please set it to a valid Hugging Face token.")

# 1. LOAD DATASET FROM HUGGING FACE
print("Loading raw dataset from Hugging Face...")
dataset = load_dataset(DATASET_REPO, token=hf_token)
df = pd.DataFrame(dataset['train'])
print("Raw dataset loaded.")

# 2. PERFORM CLEANING & STRATIFIED 80/20 SPLIT
def prepare_engine_data(df):
    """
    Cleans the engine sensor data based on EDA insights.
    """
    required_cols =['Engine rpm', 'Lub oil pressure', 'Fuel pressure',
           'Coolant pressure', 'lub oil temp', 'Coolant temp', 'Engine Condition']
    df = df[required_cols]

    df = df.drop_duplicates()

    df = df[df['Engine rpm'] >= 0]

    return df

print("Cleaning data...")
df_cleaned = prepare_engine_data(df)
print("Data cleaned.")

print("Performing stratified train-test split...")
train_df, test_df = train_test_split(
    df_cleaned,
    test_size=0.20,
    random_state=42,
    stratify=df_cleaned['Engine Condition']
)
print("Data split into train and test sets.")

# SAVE LOCALLY
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
print("Train and test datasets saved locally.")

# 3. UPLOAD BACK TO HUGGING FACE
print("Uploading processed datasets to Hugging Face...")
api = HfApi(token=hf_token)

# Upload Train
api.upload_file(
    path_or_fileobj="train.csv",
    path_in_repo="data/train.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset",
    commit_message="Add processed train.csv"
)

# Upload Test
api.upload_file(
    path_or_fileobj="test.csv",
    path_in_repo="data/test.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset",
    commit_message="Add processed test.csv"
)

print("Processed datasets uploaded to Hugging Face.")

# Clean up local files
os.remove("train.csv")
os.remove("test.csv")
print("Local temporary files removed.")
