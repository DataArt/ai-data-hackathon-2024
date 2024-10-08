from typing import List
import joblib
import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import matplotlib.pyplot as plt

import pandas as pd
import pandasql as ps
import streamlit as st

from src.model import preprocess_data


# TOOLS
@tool(parse_docstring=True)
def show_column_distribution(column: str) -> pd.Series:
    """
    Save the column distribution of the financial DataFrame as an image and return the image path.

    Args:
        column: Column name in the financial DataFrame.

    Returns:
        Path to the saved image.
    """
    global df
    
    logging.info(f"Generating distribution plot for column: {column}")
    fig, ax = plt.subplots()
    df[column].value_counts().plot(kind='bar', ax=ax)
    
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {column}')
    
    logging.info("Saving the plot as an image.")
    image_path = 'data/tmp_image.png'
    plt.savefig(image_path)
    
    logging.info("Closing the plot.")
    plt.close(fig)
    
    # Return the path to the saved image
    return image_path


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
    loaded_objects = joblib.load("src/rf_model_pipeline_compressed.pkl")
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


def get_agent(llm):
    memory = MemorySaver()
    tools = [query_financial_df, classify_fraud, show_column_distribution]
    agent_executor = create_react_agent(llm, tools, checkpointer=memory)
    return agent_executor


# DATA
df = pd.read_csv('s3://hackathon.datasets/Bank Account Fraud Dataset Suite/Base.csv', nrows=100)
df = df.drop(columns=['fraud_bool'])


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


def format_tool_decision(input_dict):
    tool_name = input_dict[0]["name"]
    arguments = input_dict[0]["args"]
    formatted_args = ",\n\t".join([f'"{k}":"{v}"' for k, v in arguments.items()])
    result = f'**🔧 The agent decided to use the tool:** `{tool_name}`\n\n**📋 With the following arguments:**\n```\n{formatted_args}\n```'
    return result


st.set_page_config(page_title="QueryShield: Chat with your data", page_icon=":shield:")
st.title(":shield: QueryShield: Chat with your data")


if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # MODEL and AGENT
    llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs=dict(temperature=0),
    )
    agent = get_agent(llm=llm)

    # CHAT
    image_generated = False
    with st.chat_message("assistant"):
        for chunk in agent.stream(
                {"messages": [HumanMessage(content=prompt)]}, config={"configurable": {"thread_id": 42}}
        ):  
            logging.info(chunk)
            if "agent" in chunk:
                agnt_msg = chunk['agent']['messages'][0]
                if agnt_msg.content:
                    st.write(agnt_msg.content)
                    st.session_state.messages.append({"role": "assistant", "content": agnt_msg.content})
                else:
                    formatted_msg = format_tool_decision(agnt_msg.tool_calls)
                    if agnt_msg.tool_calls[0]["name"] == "show_column_distribution":
                        image_generated = True
            else:
                if image_generated:
                    st.image('data/tmp_image.png')
                    image_generated = False