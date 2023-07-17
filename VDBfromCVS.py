import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# Creating the DB:
# # -----------------------------------------------
def create_vector_DB_from_csv():
    # 1. Load your model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # # 2. Load your data from the CSV file
    df = pd.read_csv('./Data/nasdaq_symbols_to_name3.csv')

    # Generate name_to_symbol and symbol_to_name dictionaries
    name_to_symbol = dict(zip(df['Company'], df['Symbol']))
    symbol_to_name = dict(zip(df['Symbol'], df['Company']))

    # Preprocess the company names
    df['Company'] = df['Company'].replace('Common Stock', '', regex=True)
    df['Company'] = df['Company'].replace('Class A', '', regex=True)
    df['Company'] = df['Company'].replace('Class B', '', regex=True)
    df['Company'] = df['Company'].replace('Class C', '', regex=True)
    df['Company'] = df['Company'].replace('PLC', '', regex=True)
    df['Company'] = df['Company'].replace('Holdings', '', regex=True)
    df['Company'] = df['Company'].replace('Inc\.', '', regex=True)
    df['Company'] = df['Company'].replace('Corp\.', '', regex=True)
    df['Company'] = df['Company'].replace('Ltd\.', '', regex=True)
    df['Company'] = df['Company'].replace('.com\.', '', regex=True)
    df['Company'] = df['Company'].replace('Opportunity', '', regex=True)
    df['Company'] = df['Company'].replace('Shares', '', regex=True)
    df['Company'] = df['Company'].str.strip()  # This line removes leading and trailing spaces


    # Print the first few lines of your DataFrame to check the columns
    print(df.head())

    # 3. Get the company names and symbols from the dataframe
    company_names = df['Company'].tolist()
    # symbols = df['Symbol'].tolist()

    # 4. Create embeddings for your data
    name_embeddings = model.encode(company_names)

    # Normalize the embeddings for cosine similarity
    # faiss.normalize_L2(name_embeddings)

    # 5. Create a new FAISS index
    dimension = name_embeddings.shape[1]  # This is the dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)

    # 6. Add vectors to the FAISS index (only company names)
    index.add(name_embeddings.astype('float32'))

    # 7. Create a mapping of index to identifier (only company names)
    identifier_mapping = {i: identifier for i, identifier in enumerate(company_names)}

    # 8. Optionally, save the index and mapping for later use
    faiss.write_index(index, './Files/faiss_index.fai')
    np.save('./Files/identifier_mapping.npy', identifier_mapping)
    print('Finished creating FAISS index!')

    # # 9. Query the index
    # query = "Apple Inc. is doing well today, and so is MSFT"
    # query_embedding = model.encode(query)
    # D, I = index.search(np.array([query_embedding]).astype('float32'), k=5)  # k is the number of nearest neighbors to return

    # Threshhold checks:
    # for threshold in np.arange(0.5, 2, 0.1):  # Testing thresholds from 0.5 to 2.0, in increments of 0.1
    #     matches = [identifier_mapping[i] for d, i in zip(D[0], I[0]) if d < threshold]
    #     print(f"Threshold: {threshold}, Matches: {matches}")

    # # # 9. Set a threshold
    # threshold = 0.9999999999999999  # or any value you consider appropriate

    # # 10. Use the identifier_mapping to find the corresponding company names or symbols
    # matches = [identifier_mapping[i] for d, i in zip(D[0], I[0]) if d < threshold]
    # print(matches)


# create_vector_DB_from_csv()