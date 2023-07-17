import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy

# # 1. Load your model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # # 2. Load your data from the CSV file
# df = pd.read_csv('nasdaq_symbols_to_name.csv')

# # Generate name_to_symbol and symbol_to_name dictionaries
# name_to_symbol = dict(zip(df['Company'], df['Symbol']))
# symbol_to_name = dict(zip(df['Symbol'], df['Company']))

# # Load the index and mapping
# index = faiss.read_index('faiss_index.fai')
# identifier_mapping = np.load('identifier_mapping.npy', allow_pickle=True).item()

# # Define your query
# # query = "Hello how are you MSFT"
# # query = "Apple Inc. is doing well today, and so is MSFT"
# # query = "Who will win Big Tech’s race for the best AI assistant: Google, Apple, Meta or Amazon?"
# # query = "Amazon"
# query = "Hello"

# # Tokenize the query into words
# words = query.split()

# # Filter out non-alphanumeric characters
# import re
# words = [re.sub(r'\W+', '', word) for word in words]

# # Separate the words into those that match exactly with your symbols and those that don't
# symbols = [word for word in words if word in symbol_to_name.keys()]
# non_symbol_words = [word for word in words if word not in symbols]

# # Create a sentence from the non-symbol words
# non_symbol_sentence = ' '.join(non_symbol_words)

# # Create an embedding for the entire non-symbol sentence
# non_symbol_sentence_embedding = model.encode([non_symbol_sentence])

# # # Normalize the query embedding for cosine similarity
# # faiss.normalize_L2(non_symbol_sentence_embedding)

# # Search the index
# D, I = index.search(np.array(non_symbol_sentence_embedding).astype('float32'), k=5)  # change k to the number of matches you want

# # 9. Set a threshold
# threshold = 1.0999999999999999 # or any value you consider appropriate

# # # 10. Use the identifier_mapping to find the corresponding company names
# matches = [identifier_mapping[i] for d, i in zip(D[0], I[0]) if d < threshold]
# print("Matched company names:", matches)
# print("Matched symbols:", symbols)

# # Threshhold checks:
# # for threshold in np.arange(0.5, 2, 0.1):  # Testing thresholds from 0.5 to 2.0, in increments of 0.1
# #     matches = [identifier_mapping[i] for d, i in zip(D[0], I[0]) if d < threshold]
# #     print(f"Threshold: {threshold}, Matches: {matches}")
# #     print(f"Threshold: {threshold} Symbols: {symbols}")

# ----------------------------------------------------------- Version 2

nlp = spacy.load("en_core_web_sm")

# 1. Load your model
model = SentenceTransformer('all-MiniLM-L6-v2')

# # 2. Load your data from the CSV file
df = pd.read_csv('nasdaq_symbols_to_name.csv')

# Generate name_to_symbol and symbol_to_name dictionaries
name_to_symbol = dict(zip(df['Company'], df['Symbol']))
symbol_to_name = dict(zip(df['Symbol'], df['Company']))

# Load the index and mapping
index = faiss.read_index('faiss_index.fai')
identifier_mapping = np.load('identifier_mapping.npy', allow_pickle=True).item()

# Define your query
# query = "Hello how are you MSFT"
# query = "Apple Inc. is doing well today, and so is MSFT"
# query = "Who will win Big Tech’s race for the best AI assistant: Google, Apple, Meta or Amazon?"
# query = "Amazon"
# query = "Hello"
# query = "Why I Never Buy a New Car: Warren Buffet - New Trader U"
query = "Microsoft bought openAI for a few million dollars now google and meta are joining the fight over next gen AI"


# Tokenize the query into words
words = query.split()

# Check for exact matches in symbols
symbols = [word for word in words if word in symbol_to_name]

# Remove symbols from the list of words
non_symbol_words = [word for word in words if word not in symbols]

# Now, perform semantic search on each non-symbol word individually

# Collect matches for all non-symbol words
all_matches = []

if non_symbol_words:
    # Create embeddings for all non-symbol words
    non_symbol_word_embeddings = model.encode(non_symbol_words)

    # Search the index
    D, I = index.search(np.array(non_symbol_word_embeddings).astype('float32'), k=5)  # k is the number of nearest neighbors to return

    # Set a fixed threshold
    threshold = 0.5  # Adjust this value as needed

    # Iterate over all results
    for d_list, i_list in zip(D, I):
        # Use the identifier_mapping to find the corresponding company names
        matches = [identifier_mapping[i] for d, i in zip(d_list, i_list) if d < threshold]
        all_matches.extend(matches)

print("Matched company names:", all_matches)
print("Matched symbols:", symbols)

# # -------------------------------------------------- Version 3

# import spacy
# import numpy as np

# # 1. Load your model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # 2. Load your data from the CSV file
# df = pd.read_csv('nasdaq_symbols_to_name.csv')

# # Generate name_to_symbol and symbol_to_name dictionaries
# name_to_symbol = dict(zip(df['Company'], df['Symbol']))
# symbol_to_name = dict(zip(df['Symbol'], df['Company']))

# # Create additional dictionary for checking company names
# name_keys = name_to_symbol.keys()

# # Load the index and mapping
# index = faiss.read_index('faiss_index.fai')
# identifier_mapping = np.load('identifier_mapping.npy', allow_pickle=True).item()

# # Define your query
# # query = "Microsoft bought openAI for a few million dollars now google and meta are joining the fight over next gen AI"
# # # query = "Hello how are you MSFT"
# # # query = "Apple Inc. is doing well today, and so is MSFT"
# # # query = "Who will win Big Tech’s race for the best AI assistant: Google, Apple, Meta or Amazon?"
# query = "Amazon"
# # query = "Hello"

# # Tokenize the query into words
# words = query.split()

# # Load the spaCy model
# nlp = spacy.load('en_core_web_sm')

# # Parse the text with spaCy. This runs the entire pipeline.
# doc = nlp(query)

# # 'doc' now contains a parsed version of text. We can use it to do anything we want!
# # For example, this will print out all the named entities that were detected:
# for entity in doc.ents:
#     print(entity.text, entity.label_)

# # Filter out non-alphanumeric characters
# import re
# query = re.sub(r'\W+', ' ', query)

# # Tokenize the query into words
# words = query.split()

# # Separate the words into those that match exactly with your symbols and those that don't
# symbols = [word for word in words if word in symbol_to_name.keys()]
# non_symbol_words = [word for word in words if word not in symbols]

# # Create a sentence from the non-symbol words
# non_symbol_sentence = ' '.join(non_symbol_words)

# # Create embeddings for the proper nouns
# proper_noun_embeddings = []
# for noun_phrase in doc.noun_chunks:  # for each noun phrase in the text
#     if len(noun_phrase.text.split()) > 1 or noun_phrase.text in name_keys or noun_phrase.text in symbol_to_name.keys():  
#         # Don't skip single-word noun phrases that are known company names or symbols
#         np_embedding = model.encode([noun_phrase.text])
#         proper_noun_embeddings.append(np_embedding)

# # Check if proper_noun_embeddings is not empty
# if proper_noun_embeddings:
#     # Stack all embeddings into a numpy array
#     proper_noun_embeddings = np.vstack(proper_noun_embeddings)

#     # Normalize the embeddings for cosine similarity
#     faiss.normalize_L2(proper_noun_embeddings)

#     # Search the index
#     D, I = index.search(proper_noun_embeddings, k=5)  # change k to the number of matches you want

#     # Set a threshold
#     threshold = 0.9 # or any value you consider appropriate

#     #  Threshhold checks:
#     for threshold in np.arange(0.5, 2, 0.1):  # Testing thresholds from 0.5 to 2.0, in increments of 0.1
#         matches = [identifier_mapping[i] for d, i in zip(D[0], I[0]) if d < threshold]
#         print(f"Threshold: {threshold}, Matches: {matches}")
#         print(f"Threshold: {threshold}, Symbols: {symbols}")

#     # # Use the identifier_mapping to find the corresponding company names
#     # matches = [identifier_mapping[i] for d, i in zip(D[0], I[0]) if d < threshold]
# else:
#     matches = []

# print("Matched company names:", matches)
# print("Matched symbols:", symbols)
