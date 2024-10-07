import logging
import pandas as pd
import joblib  # For saving and loading models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load data and pre-process it
def load_and_preprocess_data(data_path):
    logging.info("Loading dataset from: %s", data_path)
    df = pd.read_csv(data_path)
    
    # Drop unnecessary columns
    df = df.drop(['device_fraud_count'], axis=1, errors='ignore')

    # Split data into features and target
    X = df.drop(['fraud_bool'], axis=1)
    y = df['fraud_bool']

    return X, y

# Function for pre-processing categorical and numerical features
def preprocess_data(X, onehot_encoder=None, scaler=None, is_train=True):
    # Identify object columns
    object_cols = X.select_dtypes(include=['object']).columns.tolist()

    # One-Hot Encoding for categorical features
    if is_train:
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe_cols = pd.DataFrame(onehot_encoder.fit_transform(X[object_cols]))
    else:
        ohe_cols = pd.DataFrame(onehot_encoder.transform(X[object_cols]))

    ohe_cols.index = X.index  # Keep index consistent with original data

    # Remove object columns and add one-hot-encoded columns
    X_num = X.drop(object_cols, axis=1)
    X_preprocessed = pd.concat([X_num, ohe_cols], axis=1)

    # Convert column names to strings
    X_preprocessed.columns = X_preprocessed.columns.astype(str)

    # Scale numerical data
    if is_train:
        scaler = StandardScaler()
        X_preprocessed = scaler.fit_transform(X_preprocessed)
    else:
        X_preprocessed = scaler.transform(X_preprocessed)

    return X_preprocessed, onehot_encoder, scaler

# Function to train and evaluate the model
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    logging.info("Training RandomForestClassifier model.")
    rf_model = RandomForestClassifier(class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    logging.info("Evaluating model on test set.")
    y_pred = rf_model.predict(X_test)
    
    logging.info("Model performance:\n%s", classification_report(y_test, y_pred))
    auc_score = roc_auc_score(y_test, y_pred)
    logging.info("ROC AUC Score: %f", auc_score)
    
    return rf_model

# Save the model and pre-processing objects using joblib with compression
def save_model(model, onehot_encoder, scaler, model_path):
    logging.info("Saving model and pre-processing objects to %s with compression", model_path)
    joblib.dump({'model': model, 'onehot_encoder': onehot_encoder, 'scaler': scaler}, model_path, compress=3)

# Load the model and pre-processing objects for prediction
def load_model(model_path):
    logging.info("Loading model and pre-processing objects from %s", model_path)
    return joblib.load(model_path)

# Main script execution
if __name__ == "__main__":
    # Load and split the data
    data_path = 's3://hackathon.datasets/Bank Account Fraud Dataset Suite/Base.csv'
    X, y = load_and_preprocess_data(data_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the training and test data
    X_train_preprocessed, onehot_encoder, scaler = preprocess_data(X_train, is_train=True)
    X_test_preprocessed, _, _ = preprocess_data(X_test, onehot_encoder=onehot_encoder, scaler=scaler, is_train=False)

    # Train and evaluate the model
    rf_model = train_and_evaluate_model(X_train_preprocessed, y_train, X_test_preprocessed, y_test)

    # Save the trained model and pre-processing objects
    save_model(rf_model, onehot_encoder, scaler, "rf_model_pipeline_compressed.pkl")

    # Load the model and pre-processing pipeline from disk
    loaded_objects = load_model("rf_model_pipeline_compressed.pkl")
    loaded_model = loaded_objects['model']
    loaded_onehot_encoder = loaded_objects['onehot_encoder']
    loaded_scaler = loaded_objects['scaler']

    # Ensure we get the same metrics by re-evaluating the loaded model
    logging.info("Re-evaluating the loaded model.")
    X_test_preprocessed_loaded, _, _ = preprocess_data(X_test, onehot_encoder=loaded_onehot_encoder, scaler=loaded_scaler, is_train=False)

    y_pred_loaded = loaded_model.predict(X_test_preprocessed_loaded)

    logging.info("Loaded model performance:\n%s", classification_report(y_test, y_pred_loaded))
    auc_score_loaded = roc_auc_score(y_test, y_pred_loaded)
    logging.info("Loaded model ROC AUC Score: %f", auc_score_loaded)

    # Check if the metrics match
    if auc_score_loaded:
        logging.info("Model successfully loaded and re-evaluated with the same performance.")
    else:
        logging.warning("There was an issue with loading and re-evaluating the model.")