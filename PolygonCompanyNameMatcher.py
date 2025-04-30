import requests
import pandas as pd
import csv
import time
import os
from dotenv import load_dotenv


import logging

# Set up basic logging
logging.basicConfig(filename='polygon_stocks.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Load environment variables
load_dotenv()

class PolygonStocks:
    def __init__(self, api_key, old_stocks_filepath):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.old_stocks_filepath = old_stocks_filepath

    def load_existing_data(self):
        try:
            return pd.read_csv(self.old_stocks_filepath)
        except FileNotFoundError:
            print(f"File not found: {self.old_stocks_filepath}")
            return pd.DataFrame()

    def list_stocks(self, active=True):
        start_time = time.time()  # Start the timer
        endpoint = f"{self.base_url}/v3/reference/tickers"
        params = {
            "apiKey": self.api_key,
            "active": str(active).lower(),
            "asset_class": "stocks",
            "market": "stocks"
        }
        all_stocks = []  # List to accumulate all stocks

        try:
            while endpoint:
                response = requests.get(endpoint, params=params)
                response.raise_for_status()  # Raises HTTPError for bad responses
                data = response.json()
                all_stocks.extend(data.get('results', []))  # Add results to all_stocks list

                # Polygon.io provides the full URL for the next page in 'next_url'
                endpoint = data.get('next_url', None)
                # params = {}  # Clear params since 'next_url' will include necessary parameters

        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP Error while fetching stocks: {e}")
        except KeyError:
            logging.error("Unexpected response structure while fetching stocks")

        elapsed_time = time.time() - start_time  # Calculate elapsed time
        print(f"Fetching all stocks took {elapsed_time:.2f} seconds.")
        logging.info(f"Fetching all stocks took {elapsed_time:.2f} seconds.")

        return all_stocks

    def fetch_stock_events_and_name(self, symbol):
        endpoint = f"{self.base_url}/vX/reference/tickers/{symbol}/events"
        params = {"apiKey": self.api_key}
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            print(f"Data for {symbol}: {data}")  # For debugging
            if 'results' in data and isinstance(data['results'], dict):
                events = data['results'].get('events', [])
                name = data['results'].get('name', 'Unknown')  # Extract name
                return events, name
            else:
                logging.warning(f"No events or name found for {symbol}.")
                return [], 'Unknown'
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP Error while fetching events for {symbol}: {e}")
            return [], 'Unknown'
        except KeyError as ke:
            logging.error(f"KeyError encountered while fetching events for {symbol}: {ke}")
            return [], 'Unknown'



    def analyze_stock_events(self, events):
        analyzed_events = []  # A list to hold dictionaries of analyzed event information

        for event in events:
            event_info = {}
            if event['type'] == 'ticker_change':
                event_info['type'] = 'Ticker Change'
                event_info['new_ticker'] = event['ticker_change']['ticker']
                event_info['date'] = event['date']
                analyzed_events.append(event_info)
            # Add additional event types as needed
        return analyzed_events

    def format_polygon_data(self, polygon_data):
        polygon_data.rename(columns={'ticker': 'Symbol', 'name': 'Company'}, inplace=True)
        # Add further formatting steps as necessary
        return polygon_data
    
    def update_ticker_transitions(self):
        existing_data_df = self.load_existing_data()
        updated_data_df = existing_data_df.copy()

        # Step 1: Fetch all tickers
        all_tickers = set(existing_data_df['Symbol'])

        # Step 2 & 3: Process events to deduce ticker transitions
        for ticker in all_tickers:
            events = self.fetch_stock_events(ticker)
            sorted_events = sorted(events, key=lambda x: x.get('date', ''))

            # Process each event in chronological order
            for i, event in enumerate(sorted_events):
                if event['type'] == 'ticker_change':
                    new_ticker = event['ticker_change']['ticker']
                    # If it's the first event, the old ticker is the one we're currently processing
                    if i == 0:
                        old_ticker = ticker
                    else:
                        # Otherwise, the old ticker is the new ticker from the previous event
                        old_ticker = sorted_events[i-1]['ticker_change']['ticker']

                    # Update the dataset: Replace old ticker with new one
                    if old_ticker in updated_data_df['Symbol'].values:
                        updated_data_df.loc[updated_data_df['Symbol'] == old_ticker, 'Symbol'] = new_ticker
                        logging.info(f"Updated ticker: {old_ticker} to {new_ticker}")

        # Step 4: Save the updated dataset
        updated_data_df.to_csv(self.old_stocks_filepath, index=False)
        logging.info("Ticker transitions updated and saved.")


    def check_and_update_tickers(self, new_data):
        # Step 1: Load new data and existing dataset
        new_stocks_df = pd.DataFrame(new_data)
        existing_data_df = self.load_existing_data()

        # Step 2: Identify new tickers not already in the existing dataset
        new_tickers = set(new_stocks_df['ticker']) - set(existing_data_df['Symbol'])
        new_symbols_info = []

        # Step 3: Fetch events and names for each new ticker
        for ticker in new_tickers:
            events, name = self.fetch_stock_events_and_name(ticker)

            # Assume the ticker remains unchanged unless events indicate otherwise
            new_ticker = ticker
            if events:
                # Sort events by date to ensure they are processed in chronological order
                sorted_events = sorted(events, key=lambda x: x.get('date', ''))
                latest_event = sorted_events[-1]
                if latest_event['type'] == 'ticker_change':
                    # If the latest event is a ticker change, update the new_ticker variable
                    old_ticker = latest_event['ticker_change']['ticker']
                    new_ticker = ticker

            # Compile symbol information including the potentially updated ticker and its name
            symbol_info = {'Symbol': new_ticker, 'Name': name, "OldTicker": old_ticker}
            new_symbols_info.append(symbol_info)

        # Step 4: Integrate new symbols into the dataset
        if new_symbols_info:
            # Efficiently add all new symbols to the dataset in a single operation
            new_rows_df = pd.DataFrame(new_symbols_info)
            existing_data_df = pd.concat([existing_data_df, new_rows_df], ignore_index=True)
            
            # Step 5: Save the updated dataset
            existing_data_df.to_csv(self.old_stocks_filepath, index=False)
            logging.info("Stock data update complete with new symbols and their additional info.")



        
    def run_update(self):
        current_stocks_data = self.list_stocks()
        self.check_and_update_tickers(current_stocks_data)
        print(f"Stock data has been updated from Polygon.io and saved to {self.old_stocks_filepath}")

# Example usage:
polygon_api_key = os.environ.get("POLYGON_API_KEY")
old_stocks_filepath = 'Data/nasdaq_symbols_to_name3_test.csv'
polygon_stocks = PolygonStocks(polygon_api_key, old_stocks_filepath)
polygon_stocks.run_update()
