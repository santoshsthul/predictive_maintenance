
import pandas as pd
import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from datasets import load_dataset
import os

# Configuration
DATASET_REPO = "SantoshS23/PredMaintDataSet"
MODEL_REPO_ID = "SantoshS23/PredMaintModel"
MODEL_REPO_TYPE = "model"
MODEL_FILENAME = "adaboost_predictive_maintenance_model.joblib"

# Get Hugging Face token from environment variables
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set. Please set it to a valid Hugging Face token.")

# 1. LOAD DATA FROM HUGGING FACE
print("Loading data from Hugging Face...")
dataset = load_dataset(
    path=DATASET_REPO,
    data_files={'train': 'data/train.csv', 'test': 'data/test.csv'},
    token=hf_token
)

train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

X_train = train_df.drop(columns=['Engine Condition'])
y_train = train_df['Engine Condition']
X_test = test_df.drop(columns=['Engine Condition'])
y_test = test_df['Engine Condition']

print("Data loaded and split.")

# 2. DEFINE AND TRAIN THE BEST MODEL (AdaBoost)
# Using the best parameters found during hyperparameter tuning
print("Initializing and training AdaBoost model...")
final_model = AdaBoostClassifier(learning_rate=0.5, n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)
print("Model training complete.")

# 3. EVALUATE FINAL MODEL
print("Evaluating final model...")
y_pred_final = final_model.predict(X_test)

report = classification_report(y_test, y_pred_final)
accuracy = accuracy_score(y_test, y_pred_final)
roc_auc = roc_auc_score(y_test, y_pred_final)
f1_class_1 = f1_score(y_test, y_pred_final, pos_label=1)

print("Final Model Performance (AdaBoost):")
print(report)
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score (Faulty - Class 1): {f1_class_1:.4f}")

# SAVE AND REGISTER THE BEST MODEL TO HUGGING FACE
print("Saving and registering model to Hugging Face...")
model_path_local = MODEL_FILENAME
joblib.dump(final_model, model_path_local)

api = HfApi(token=hf_token)

# Create the model repo if it doesn't exist
try:
    api.repo_info(repo_id=MODEL_REPO_ID, repo_type=MODEL_REPO_TYPE)
    print(f"Model space '{MODEL_REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model space '{MODEL_REPO_ID}' not found. Creating new model space...")
    create_repo(repo_id=MODEL_REPO_ID, repo_type=MODEL_REPO_TYPE, private=False, token=hf_token)
    print(f"Model space '{MODEL_REPO_ID}' created.")

# Upload the model file
api.upload_file(
    path_or_fileobj=model_path_local,
    path_in_repo=MODEL_FILENAME,
    repo_id=MODEL_REPO_ID,
    repo_type=MODEL_REPO_TYPE,
    commit_message=f"Upload AdaBoost model with F1 (Class 1): {f1_class_1:.4f}"
)

print(f"Best model '{MODEL_FILENAME}' registered on Hugging Face Model Hub with F1 (Class 1): {f1_class_1:.4f}")

# Clean up local model file
os.remove(model_path_local)
print("Local model file removed.")
