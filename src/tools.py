from typing import List
import pandasql as ps
import random
from langchain_core.tools import tool
import logging
import pandas as pd
import joblib
from model import preprocess_data 

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@tool(parse_docstring=True)
def show_column_distribution(column: str) -> pd.Series:
    """
    Print image of the column distribution of the financial DataFrame

    Args:
        column: Column name in the financial DataFrame.
    """
    global df
    plot = df[column].value_counts().plot(kind='bar')
    plot.get_figure()


@tool(parse_docstring=True)
def classify_fraud(query: str):
    """
    Classify rows of the df whether they are fraud or not.

    Args:
        query: SQL query string to execute on the DataFrame to obtain df rows to classify
    """
    input_df = ps.sqldf(query)

    # Load the saved model and pre-processing objects from the disk
    logging.info("Loading model and pre-processing pipeline from: %s")
    loaded_objects = joblib.load("rf_model_pipeline_compressed.pkl")
    loaded_model = loaded_objects['model']
    loaded_onehot_encoder = loaded_objects['onehot_encoder']
    loaded_scaler = loaded_objects['scaler']

    # Drop any columns that were not used in the model training
    logging.info("Dropping columns that were not used in training.")
    input_df = input_df.drop(['device_fraud_count', 'fraud_bool'], axis=1, errors='ignore')

    # Apply the same pre-processing to the input data
    logging.info("Applying pre-processing to the input data.")
    X_preprocessed, _, _ = preprocess_data(input_df, onehot_encoder=loaded_onehot_encoder, scaler=loaded_scaler, is_train=False)

    # Make predictions using the loaded model
    logging.info("Making predictions.")
    predictions = loaded_model.predict(X_preprocessed)

    return predictions

    
@tool(parse_docstring=True)
def query_financial_df(query: str):
    """
    Queries the financial DataFrame using SQL-like syntax.
    The DataFrame contains the following columns:
        fraud_bool (int64): Indicator of fraud.
        income (float64): Reported income.
        name_email_similarity (float64): Similarity between name and email.
        prev_address_months_count (int64): Months at the previous address.
        current_address_months_count (int64): Months at the current address.
        customer_age (int64): Age of the customer.
        days_since_request (float64): Days since the request.
        intended_balcon_amount (float64): Intended balance amount.
        payment_type (object): Type of payment.
        zip_count_4w (int64): Zip code count in the last 4 weeks.
        velocity_6h (float64): Transaction velocity in the last 6 hours.
        velocity_24h (float64): Transaction velocity in the last 24 hours.
        velocity_4w (float64): Transaction velocity in the last 4 weeks.
        bank_branch_count_8w (int64): Bank branch count in the last 8 weeks.
        date_of_birth_distinct_emails_4w (int64): Number of distinct emails in the last 4 weeks related to date of birth.
        employment_status (object): Employment status of the customer.
        credit_risk_score (int64): Credit risk score of the customer.
        email_is_free (int64): Indicator if the email is from a free provider.
        housing_status (object): Housing status of the customer.
        phone_home_valid (int64): Indicator if the home phone is valid.
        phone_mobile_valid (int64): Indicator if the mobile phone is valid.
        bank_months_count (int64): Months with the bank.
        has_other_cards (int64): Indicator if the customer has other cards.
        proposed_credit_limit (float64): Proposed credit limit.
        foreign_request (int64): Indicator if the request is foreign.
        source (object): Source of the request.
        session_length_in_minutes (float64): Length of the session in minutes.
        device_os (object): Operating system of the device.
        keep_alive_session (int64): Indicator if the session is kept alive.
        device_distinct_emails_8w (int64): Distinct emails on the device in the last 8 weeks.
        device_fraud_count (int64): Fraud count associated with the device.
        month (int64): Month of the request.
    Example queries:
        SELECT * FROM df WHERE income > 0.2
        SELECT * FROM df ORDER BY "index" DESC LIMIT 10

    Args:
        query: SQL query string to execute on the DataFrame.
    """
    res_df = ps.sqldf(query)
    return res_df if len(res_df) < 10 else res_df.head(10)


# test predict in main: predict for first 5 rows
if __name__ == "__main__":
    # Load and split the data
    data_path = 's3://hackathon.datasets/Bank Account Fraud Dataset Suite/Base.csv'
    df = pd.read_csv(data_path)
    query = "SELECT * FROM df LIMIT 5"
    print(classify_fraud(query))