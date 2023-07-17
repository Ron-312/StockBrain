import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy

# Load your model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load your data from the CSV file
df = pd.read_csv('./Data/nasdaq_symbols_to_name3.csv')

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

# Generate name_to_symbol and symbol_to_name dictionaries
name_to_symbol = dict(zip(df['Company'], df['Symbol']))
symbol_to_name = dict(zip(df['Symbol'], df['Company']))

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

# -------------------- Semantic queries on the Vector DB ----------------------

# Check if company name exists in tweet

def check_if_company_exist_in_tweet(query):

    # Load the index and mapping
    index = faiss.read_index('./Files/faiss_index.fai')
    identifier_mapping = np.load('./Files/identifier_mapping.npy', allow_pickle=True).item()

    # Define your query, examples
    # query = "Hello how are you MSFT"
    # query = "Apple Inc. is doing well today, and so is MSFT"
    # query = "Who will win Big Tech’s race for the best AI assistant: Google, Apple, Meta or Amazon?"
    # query = "Meta"
    # query = "Amazon"
    # query = "Hello"
    # query = "Why I Never Buy a New Car: Warren Buffet - New Trader U"
    # query = "Is #India the new battleground for #streaming giants like $DIS and $NFLX? Jonathan Kees of Daiwa America shares his view. @TanvirGill2 $AMZN $WBD $CMCSA #earnings"
    # query = "Family offices move to 'risk on' with plans to load up on stocks, private credit"
    # query = "JPMorgan upgrades Taser maker: 'Pullback we were waiting for came sooner than expected'"

    # Remove special characters
    chars_to_remove = "@$#?!,:'\""
    trans_table = str.maketrans(chars_to_remove, ' '*len(chars_to_remove))
    query = query.translate(trans_table)

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
        threshold = 0.4  # Adjust this value as needed

        # Iterate over all results
        for d_list, i_list in zip(D, I):
            # Use the identifier_mapping to find the corresponding company names
            matches = [identifier_mapping[i] for d, i in zip(d_list, i_list) if d < threshold]
            all_matches.extend(matches)

        #     # Threshhold checks:
        # for threshold in np.arange(0.5, 2, 0.1):  # Testing thresholds from 0.5 to 2.0, in increments of 0.1
        #     matches = [identifier_mapping[i] for d, i in zip(D[0], I[0]) if d < threshold]
        #     print(f"Threshold: {threshold}, Matches: {matches}")
        #     print(f"Threshold: {threshold} Symbols: {symbols}")


    print("Matched company names:", all_matches)
    print("Matched symbols:", symbols)
    return {
        "Matched company names": all_matches,
        "Matched symbols": symbols
    }

def symbol_per_company_name(company_name, sentence, thresholds=np.arange(0.5, 5, 0.3)):
    
    # Remove special characters from the senteQnce
    chars_to_remove = "@$#?!,:'\""
    trans_table = str.maketrans(chars_to_remove, ' '*len(chars_to_remove))
    sentence = sentence.translate(trans_table)
    company_name = company_name.translate(trans_table)

    # Split the company_name into words
    company_words = company_name.lower().split()

    # If the company_name is actually a symbol in the list, return it directly
    if company_name.upper() in symbol_to_name.keys():
        print(f"Matched symbol: {company_name.upper()}")
        return company_name.upper()

    # Tokenize the sentence
    sentence_words = sentence.split()

    # Check if any word in the sentence is a symbol and collect potential matches
    potential_symbols = [word for word in sentence_words if word in symbol_to_name.keys() and all(c_word in symbol_to_name[word].lower() for c_word in company_words)]

    # If there's only one potential symbol, return it
    if len(potential_symbols) == 1:
        print(f"Matched symbol: {potential_symbols[0]}")
        return potential_symbols[0]

    # If there are multiple potential symbols, perform a semantic search
    if len(potential_symbols) > 1:

        # Generate the sentence context for the input company name
        input_context = company_name + ' ' + sentence
        input_embedding = model.encode([input_context])

        # Prepare the sentence contexts and embeddings for the potential matches
        potential_contexts = [symbol_to_name[symbol] + ' ' + sentence for symbol in potential_symbols]
        potential_embeddings = model.encode(potential_contexts)

        # Calculate cosine similarities
        similarities = np.dot(input_embedding, potential_embeddings.T) / (np.linalg.norm(input_embedding) * np.linalg.norm(potential_embeddings, axis=1))

        # Find the symbol with the highest similarity
        best_matching_symbol = potential_symbols[np.argmax(similarities)]
        print(f"Matched symbol: {best_matching_symbol}")
        return best_matching_symbol

    # Tokenize the sentence
    sentence_words = sentence.lower().split()

    # Concatenate company name and sentence
    context_sentence = company_name + ' ' + sentence

    # Count the number of words in the company_name
    num_words = len(company_name.split())

    # Create a context window around the company_name in the sentence
    doc = nlp(context_sentence)

    # If the number of words is greater than the length of the sentence, reduce it
    if num_words > len(doc):
        num_words = len(doc)
    
    # Create a list to hold the context windows
    context_windows = []

    # Loop through the document to create the context windows
    for i in range(num_words, len(doc)+1):
        # Get the context window
        window = doc[i-num_words:i]

        # Convert the window to text and add to the list
        context_windows.append(window.text)

    # Encode the context windows
    context_sentence_embedding = model.encode(context_windows)

    # Filter out company names that contain any of the words in company_name
    companies = [name for name in name_to_symbol.keys() if any(word in name.lower() for word in company_words)]


    # If no companies are found, return None
    if not companies:
        print(f"No companies found for {company_name}.")
        return None

    # Create a FAISS index for semantic search
    index = faiss.IndexFlatL2(context_sentence_embedding.shape[1])

    # Prepare embeddings for all candidate companies with the sentence context
    candidate_embeddings = [model.encode([candidate + ' ' + sentence]) for candidate in companies]

    # Add the embeddings to the index
    index.add(np.array(candidate_embeddings).reshape(-1, context_sentence_embedding.shape[1]))

    # Perform a search
    D, I = index.search(context_sentence_embedding.astype('float32'), len(companies))

    # Initialize best match variable
    best_matching_symbol = None
    best_threshold = None

    # Iterate over thresholds
    for threshold in thresholds:
        # Check if the smallest distance is below the threshold
        if D[0][0] > threshold:
            print(f"Threshold: {threshold}, No match found.")
            continue
        else:
            # The result I[0][0] is the index of the best match in the companies list
            best_matching_company = companies[I[0][0]]

            # Get the symbol for the best matching company name
            best_matching_symbol = name_to_symbol[best_matching_company]

            # Keep track of the best threshold
            best_threshold = threshold

            print(f"Threshold: {threshold}, Matched symbol: {best_matching_symbol}")
            # Break the loop as soon as we find a match. We only want the first match which is the strictest criteria
            break  

    # If no matching symbol was found at any threshold
    if not best_matching_symbol:
        print("No matching symbol found at any threshold")
        return None

    print(f"Best threshold: {best_threshold}, Matched symbol: {best_matching_symbol}")
    return best_matching_symbol

# symbol_per_company_name("Apple","Apple is in the running for the best AI assistant, but can they pull ahead of the competition?")
# check_if_company_exist_in_tweet("Is #India the new battleground for #streaming giants like $DIS and $NFLX? Jonathan Kees of Daiwa America shares his view. @TanvirGill2 $AMZN $WBD $CMCSA #earnings")
# check_if_company_exist_in_tweet("JPMorgan upgrades Taser maker: 'Pullback we were waiting for came sooner than expected'")
# check_if_company_exist_in_tweet("'Regulators should look at a short-selling ban on banks, JPMorgan CEO Jamie Dimon says\n\nThey should go after them, and vigorously, he tells @flacqua@flacqua https://bloom.bg/42JMiFlhttps://'")
# check_if_company_exist_in_tweet("'Regulators should look at a short-selling ban on banks, $$CVNA CEO Jamie Dimon says\n\nThey should go after them, and vigorously, he tells @flacqua@flacqua https://bloom.bg/42JMiFlhttps://'")
# check_if_company_exist_in_tweet("Meta")
# symbol_per_company_name("Unity",'"For the balance of 2023, we expect revenue to grow faster than the markets in which we compete, with steady and meaningful continued progress on profitability," the company said in a letter to shareholders.$U')
# symbol_per_company_name("Unity Software","For the balance of 2023, we expect revenue to grow faster than the markets in which we compete, with steady and meaningful continued progress on profitability, the company said in a letter to shareholders.\n\n$U")
# symbol_per_company_name("Unity Technologies","For the balance of 2023, we expect revenue to grow faster than the markets in which we compete, with steady and meaningful continued progress on profitability, the company said in a letter to shareholders.\n\n$U")
symbol_per_company_name("$CEEK VR","Join us tomorrow for an exclusive interview with the minds behind the innovative CEEK VR  Don t miss out on this opportunity to learn more about this exciting technology.")