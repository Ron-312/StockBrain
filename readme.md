# StockBrain - A Flask Application for Semantic Stock Search

StockBrain is a Flask web application that creates a vector database from a CSV file. It performs semantic search on tweets and other lines of text to identify references to companies listed on the stock market. This facilitates efficient and meaningful queries, allowing users to quickly identify the stocks being discussed in a body of text.

## How do I get set up?

In the root directory:

1. Install required Python packages:

    ```
    pip install -r requirements.txt
    ```

2. The application expects the existence of certain Python modules: `VDBfromCVS` and `semanticQueriesVDB`. Ensure they are available in your Python environment.

## Running the Application

### Run with:

    ```
    python main.py
    ```

### This command will start the Flask server.

## Available Endpoints:
    POST /create_db
# This endpoint creates a vector database from a CSV file. It uses the create_vector_DB_from_csv function from the VDBfromCVS module.

    POST /query

# This endpoint performs a semantic search to identify references to stock companies in a given text (like a tweet). It accepts a JSON payload with a field named 'text' containing the text to be analyzed. It returns a JSON response containing the matched company names and stock symbols. It primarily uses the `check_if_company_exist_in_tweet` function from the `semanticQueriesVDB` module.

    POST /symbol

# This endpoint is used to find the stock symbol for a given company name in the context of a certain sentence. It accepts a JSON payload with two fields: 'company_name' and 'sentence'. It returns a JSON response containing the matched stock symbol. It primarily uses the `symbol_per_company_name` function from the `semanticQueriesVDB` module.





## Semantic Search
StockBrain utilizes semantic search to identify references to stock companies in lines of text, such as tweets. It uses sentence transformers and vector databases to understand the semantic content of the text and to facilitate efficient search operations. This enables the application to make connections between different stocks based on the semantic content of their descriptions, providing a powerful tool for stock market analysis.

### First Note
Ensure that the CSV file is correctly formatted and available for the application to read. The CSV file should contain information about all the stocks in the market and their respective symbols. The exact requirements for the CSV file depend on the implementation in VDBfromCVS.

#### Second Note: This application should be used responsibly and thoroughly tested in a controlled environment before live usage.