<img align="right" width="25%" src="https://natwest.gitlab-dedicated.com/natwestgroup/DigitalX/Technology/EnterpriseEngineering/OSPO/ospo-mascot-and-design-resources/-/raw/main/OSPO%20Mascot/live-mascot/live-ospo-mascot.png?ref_type=heads">


# QueryShield

**QueryShield** is a quick solution for **NatWest Hack Bake Off 2024** made by **DataArt**

Our current solution flow is designed to efficiently transform user queries into actionable insights. It begins with the user inputting a prompt through a Streamlit interface. This prompt is processed by an intelligent agent powered by AWS Bedrock using Claude V3, a sophisticated Large Language Model (LLM) for reasoning and decision-making.​

The agent employs several tools to handle the query:​
- **Database Querying**: It accesses the Snowflake database to retrieve necessary data.​
- **Fraud Prediction**: The data is analyzed using a Random Forest model hosted on AWS SageMaker to predict potential fraud.​
- **Visualization**: The results are prepared for easy interpretation and presentation.​

The LLM ensures that the entire process is coherent and logical, integrating data retrieval, analysis, and visualization seamlessly.​

Finally, the processed information is transformed into an output decision, providing the user with clear and actionable insights. This architecture leverages AWS services and advanced machine learning techniques to enable effective fraud detection and decision-making.​

## Architecture Diagram
![architecture diagram](https://github.com/DataArt/ai-data-hackathon-2024/blob/master/architecture/diagram.png?raw=true)


# Installation & Run

## LLM Model
The LLM model is hosted on AWS Bedrock. To access the model, you need to have the configured aws accound with access to `anthropic.claude-3-sonnet` model

## Dataset
For this usecase we are using and opensourced [Bank Account Fraud Dataset](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022?select=Variant+I.csv) from Kaggle. For the data storage we are using Snowflake. 
- Download the dataset `Base.csv` and upload it to Snowflake into DATASETS/BASE table.
- fill in the `.env` file with the Snowflake credentials.

## Project Setup with Poetry
Install poetry: https://python-poetry.org/docs/ (might take a while)

Run the following command to initialize a Poetry project with a virtual environment:
```bash
# Install dependencies
poetry install --no-root

# Activate the virtual environment
poetry shell
```

## Run Streamlit App
```bash
# Set up .env file, aws-cli and then run the following command
streamlit run app.py
```

# Contribution

1. Fork it...
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Read our [contribution guidelines](CONTRIBUTING.md)
4. Commit your changes (`git commit -am 'Add some fooBar'`)
5. Push to the branch (`git push origin feature/fooBar`)
6. Create a new merge request...

# License

Copyright 2024 NatWest Group

Distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).