import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import faiss
import re

from CompanyNameMatcher import CompanyNameMatcher



class CompanyDBManager:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        try:
            self.existing_df = self.load_existing_data()
        except Exception as e:
            print(f"Error loading data from {csv_file_path}: {e}")
            self.existing_df = pd.DataFrame()

    def load_existing_data(self):
        """
        Load the existing company data from a CSV file into a DataFrame.
        """
        try:
            return pd.read_csv(self.csv_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The file {self.csv_file_path} was not found.")
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError("The file is empty.")
        except Exception as e:
            raise Exception(f"An error occurred while loading the data: {e}")

    def preprocess_entry(self, entry):
        """
        Preprocess a new entry to match the existing data format, focusing on the 'Company' field.
        This includes removing specific terms and trimming spaces to ensure consistency with
        the preprocessing applied in create_vector_DB_from_csv.
        """
        # Define replacements as per the preprocessing in create_vector_DB_from_csv
        try:
            replacements = {
                'Common Stock': '', 'Class A': '', 'Class B': '', 'Class C': '',
                'PLC': '', 'Holdings': '', 'Inc\.': '', 'Corp\.': '', 'Ltd\.': '',
                '.com': '', 'Opportunity': '', 'Shares': ''
            }
            company_name = entry.get('Company', '')
            for key, value in replacements.items():
                company_name = re.sub(key, value, company_name)
            entry['Company'] = company_name.strip()
            return entry
        except Exception as e:
            print(f"Error preprocessing entry: {e}")
            return entry  # Returning the entry unmodified if an error occurs

    def update_or_add_entry(self, new_entry):
        """
        Decide whether to update an existing entry or add a new one based on the new entry's details.
        """
        try:
            matcher = CompanyNameMatcher()  # Instantiate the matcher
            
            # Attempt to match by symbol first
            existing_symbol_matches = self.existing_df[self.existing_df['Symbol'] == new_entry['symbol']]
            if not existing_symbol_matches.empty:
                self.update_existing_entry(new_entry, existing_symbol_matches.index[0])
            else:
                # Before using NLP, try matching company name and IPO year exactly
                exact_matches = self.existing_df[(self.existing_df['Company'].str.lower() == new_entry['name'].lower()) & 
                                                (self.existing_df['IPO Year'].astype(str) == str(new_entry['ipoyear']).split('.')[0])]
                if len(exact_matches) == 1:
                    # Exact match found, proceed to update
                    self.update_existing_entry(new_entry, exact_matches.index[0])
                else:
                    # Proceed with NLP-based matching if no exact matches
                    best_match_index = None
                    highest_similarity = 0
                    for index, row in self.existing_df.iterrows():
                        if str(row['IPO Year']).split('.')[0] == str(new_entry['ipoyear']).split('.')[0]:  # IPO Year matches
                            similarity = matcher.is_same_company(row['Company'], new_entry['name'])
                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                best_match_index = index
                    
                    if highest_similarity > 0.7:  # Threshold for considering a match
                        self.update_existing_entry(new_entry, best_match_index)
                    else:
                        # No suitable match found; consider this a new entry
                        self.add_new_entry(new_entry)
        except Exception as e:
            print(f"Error updating or adding an entry: {e}")


    def update_existing_entry(self, new_entry, index=None):
        try:
            symbol = new_entry['symbol']  # Capture the symbol from the new entry
            if index is None:
                matching_entries = self.existing_df[self.existing_df['Symbol'] == symbol]
                if matching_entries.empty:
                    raise ValueError(f"No matching entry found for the given symbol: {symbol}")
                index = matching_entries.index[0]

            # Initialize a message list to collect update info
            updates_info = []
            
            # Handling for the "Company" field to intelligently merge words
            old_company_value = self.existing_df.at[index, 'Company']
            new_company_value = new_entry.get('name', '').strip()

            # Attempt to maintain grammatical sense by only adding new unique words that do not exist in the old name
            # This simplistic approach appends new information at the end
            # For a more sophisticated NLP approach, consider using libraries like spaCy or NLTK
            new_info = " ".join([word for word in new_company_value.split() if word not in old_company_value.split()])
            
            if new_info:  # Check if there's actually new info to add
                updated_company_value = f"{old_company_value} {new_info}".strip()
                self.existing_df.at[index, 'Company'] = updated_company_value
                updates_info.append(f"Company: '{old_company_value}' -> '{updated_company_value}'")
            
            # IPO Year Special Handling
            if 'ipoyear' in new_entry:  # Check if IPO Year is provided in the new entry
                old_ipoyear_value = str(self.existing_df.at[index, 'IPO Year']).split('.')[0]  # Standardize the existing value
                new_ipoyear_value = str(new_entry['ipoyear']).split('.')[0]  # Standardize the new value
                if old_ipoyear_value != new_ipoyear_value:
                    self.existing_df.at[index, 'IPO Year'] = new_ipoyear_value  # Update with the standardized value
                    updates_info.append(f"IPO Year: '{old_ipoyear_value}' -> '{new_ipoyear_value}'")

            # General handling for other fields with checks to skip empty updates
            field_mapping = {
                'country': 'Country',
                'industry': 'Industry',
                'sector': 'Sector',
                # 'ipoyear': 'IPO Year'  # Moved this handling above
            }

            for json_field, df_field in field_mapping.items():
                # Ensuring we only proceed if the field exists and is different
                if json_field in new_entry and new_entry[json_field] != self.existing_df.at[index, df_field]:
                    new_value = str(new_entry[json_field])
                    old_value = str(self.existing_df.at[index, df_field])
                    if new_value:  # Making sure the new value is not empty
                        self.existing_df.at[index, df_field] = new_value
                        updates_info.append(f"{df_field}: '{old_value}' -> '{new_value}'")

            if updates_info:
                print(f"Updated entry for symbol '{symbol}' at index {index} with changes:")
                for update in updates_info:
                    print(update)
            else:
                print(f"No changes made for entry with symbol '{symbol}' at index {index}.")
        except Exception as e:
            print(f"Error updating an existing entry: {e}")


    def add_new_entry(self, new_entry):
        """
        Add a new entry to the DataFrame.
        """
        try:
            new_row = pd.DataFrame([{
                'Symbol': new_entry['symbol'],
                'Company': new_entry['name'],  # Corrected from 'Name' to 'Company'
                'Country': new_entry['country'],
                'Industry': new_entry['industry'],
                'Sector': new_entry['sector'],
                'IPO Year': str(new_entry['ipoyear']).split('.')[0]  # Standardizing 'IPO Year' and ensuring it's a string
            }])
            self.existing_df = pd.concat([self.existing_df, new_row], ignore_index=True)
        except Exception as e:
            print(f"Error adding a new entry: {e}")


    def save_to_csv(self):
        """
        Save the updated DataFrame back to the CSV file.
        """
        try:
            self.existing_df.to_csv(self.csv_file_path, index=False)
        except Exception as e:
            print(f"Error saving data to CSV: {e}")

    def process_new_entries(self, new_entries_json_path):
        """
        Process a list of new entries from a JSON file.
        """
        try:
            with open(new_entries_json_path, 'r') as file:
                new_entries = json.load(file)

            for entry in new_entries:
                preprocessed_entry = self.preprocess_entry(entry)
                self.update_or_add_entry(preprocessed_entry)

            # Optionally save the updated DataFrame back to CSV
            self.save_to_csv()
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error processing new entries: {e}")
        except Exception as e:
            print(f"Unexpected error during new entries processing: {e}")

    # -------- INSERT create_vector_DB_from_csv HERE --------
    def create_vector_DB_from_csv(self):
        """
        Create a vector database from the company names in the CSV file,
        including preprocessing and dictionary creations, using 'Company' as the key attribute.
        """
        try:
            # Load your model
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Preprocess the DataFrame from the existing data attribute
            df = self.existing_df.copy()

            # Preprocessing steps specific to 'Company'
            replacements = {
                'Common Stock': '', 'Class A': '', 'Class B': '', 'Class C': '',
                'PLC': '', 'Holdings': '', 'Inc\.': '', 'Corp\.': '', 'Ltd\.': '',
                '.com': '', 'Opportunity': '', 'Shares': ''
            }
            for key, value in replacements.items():
                df['Company'] = df['Company'].replace(key, value, regex=True)
            df['Company'] = df['Company'].str.strip()

            # Generate name_to_symbol and symbol_to_name dictionaries
            name_to_symbol = dict(zip(df['Company'], df['Symbol']))
            symbol_to_name = dict(zip(df['Symbol'], df['Company']))

            # Get the company names and generate embeddings
            company_names = df['Company'].tolist()
            name_embeddings = model.encode(company_names)

            # Create a new FAISS index
            dimension = name_embeddings.shape[1]  # Dimension of the embeddings
            index = faiss.IndexFlatL2(dimension)
            index.add(name_embeddings.astype('float32'))

            # Create a mapping of index to identifier (company names)
            identifier_mapping = {i: name for i,
                                name in enumerate(company_names)}

            # Save the index and mapping for later use
            faiss.write_index(index, './Files/faiss_index.fai')
            np.save('./Files/identifier_mapping.npy', identifier_mapping)

            # Optionally, store dictionaries for further use
            self.name_to_symbol = name_to_symbol
            self.symbol_to_name = symbol_to_name

            print('Finished creating and saving FAISS index and dictionaries!')
        except Exception as e:
            print(f"Error creating vector database: {e}")

        # New method to run the entire update and vector DB creation process
    def run_update_and_create_vector_db(self, new_entries_json_path):
        """
        Orchestrates the process of updating the database with new entries and
        then creating a vector database from the updated data.
        """
        try:
            print("Starting database update process...")
            self.process_new_entries(new_entries_json_path)
            print("Database update complete. Now creating vector database...")
            self.create_vector_DB_from_csv()
            print("Vector database creation complete.")
        except Exception as e:
            print(f"Error in main function: {e}")


# # Example usage
# csv_file_path = './Data/nasdaq_symbols_to_name3.csv'
# new_entries_json_path = './Data/new_data.json'
# db_manager = CompanyDBManager(csv_file_path)
# db_manager.process_new_entries(new_entries_json_path)
# db_manager.create_vector_DB_from_csv()
