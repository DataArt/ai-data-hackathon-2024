from typing import List
import pandasql as ps
import random
from langchain_core.tools import tool
import pandas as pd


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
def classify_fraud(query: str) -> List:
    """
    Classify rows of the df whether they are fraud or not.

    Args:
        query: SQL query string to execute on the DataFrame to obtain df rows to classify
    """
    input_df = ps.sqldf(query)
    return [random.choice([0, 1]) for _ in range(len(input_df))]
    
    
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