"""
Semantic Query Engine for Vector Database of Stock Symbols and Company Names

This module provides functionality to:
1. Match company names and symbols in text using semantic search
2. Map company names to their stock symbols using context-aware matching
3. Perform similarity-based searches on a vector database of company information
"""

import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
from typing import Dict, List, Optional, Union, Tuple

# -------------------- Configuration --------------------

# Load sentence transformer model
MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

# Load spaCy model for text processing
nlp = spacy.load("en_core_web_sm")

# Data paths
DATA_PATH = './Data/nasdaq_symbols_to_name3.csv'
INDEX_PATH = './Files/faiss_index.fai'
MAPPING_PATH = './Files/identifier_mapping.npy'

# Search parameters
DEFAULT_SEARCH_THRESHOLDS = np.arange(0.5, 5, 0.3)
DEFAULT_SEMANTIC_THRESHOLD = 0.4

# -------------------- Data Loading and Preprocessing --------------------

def load_stock_data() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load and preprocess stock data from CSV.
    
    Returns:
        Tuple containing name_to_symbol and symbol_to_name dictionaries
    """
    # Load data from CSV
    df = pd.read_csv(DATA_PATH)
    
    # Preprocess company names by removing common terms
    terms_to_remove = [
        'Common Stock', 'Class A', 'Class B', 'Class C', 'PLC', 
        'Holdings', 'Inc\.', 'Corp\.', 'Ltd\.', '.com\.', 
        'Opportunity', 'Shares'
    ]
    
    for term in terms_to_remove:
        df['Company'] = df['Company'].replace(term, '', regex=True)
    
    # Remove leading and trailing spaces
    df['Company'] = df['Company'].str.strip()
    
    # Generate dictionaries for lookup
    name_to_symbol = dict(zip(df['Company'], df['Symbol']))
    symbol_to_name = dict(zip(df['Symbol'], df['Company']))
    
    return name_to_symbol, symbol_to_name

# Load dictionaries
name_to_symbol, symbol_to_name = load_stock_data()

# -------------------- Helper Functions --------------------

def clean_text(text: str) -> str:
    """
    Remove special characters from text.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text string
    """
    chars_to_remove = "@$#?!,:'\""
    trans_table = str.maketrans(chars_to_remove, ' ' * len(chars_to_remove))
    return text.translate(trans_table)

# -------------------- Main Functions --------------------

def check_if_company_exist_in_tweet(query: str) -> Dict[str, List[str]]:
    """
    Check if any company names or stock symbols exist in the given text.
    
    Args:
        query: Text to analyze for company mentions
        
    Returns:
        Dictionary containing matched company names and symbols
    """
    # Load the index and mapping
    index = faiss.read_index(INDEX_PATH)
    identifier_mapping = np.load(MAPPING_PATH, allow_pickle=True).item()
    
    # Clean input text
    query = clean_text(query)
    
    # Tokenize the query into words
    words = query.split()
    
    # Check for exact matches in symbols
    symbols = [word for word in words if word in symbol_to_name]
    
    # Remove symbols from the list of words
    non_symbol_words = [word for word in words if word not in symbols]
    
    # Collect matches for all non-symbol words
    all_matches = []
    
    if non_symbol_words:
        # Create embeddings for all non-symbol words
        non_symbol_word_embeddings = model.encode(non_symbol_words)
        
        # Search the index (k=5 nearest neighbors)
        D, I = index.search(np.array(non_symbol_word_embeddings).astype('float32'), k=5)
        
        # Process results using threshold
        for d_list, i_list in zip(D, I):
            # Use the identifier_mapping to find the corresponding company names
            matches = [identifier_mapping[i] for d, i in zip(d_list, i_list) 
                       if d < DEFAULT_SEMANTIC_THRESHOLD]
            all_matches.extend(matches)
    
    print("Matched company names:", all_matches)
    print("Matched symbols:", symbols)
    
    return {
        "Matched company names": all_matches,
        "Matched symbols": symbols
    }


def symbol_per_company_name(company_name: str, sentence: str, 
                           thresholds: np.ndarray = DEFAULT_SEARCH_THRESHOLDS) -> Optional[str]:
    """
    Find the stock symbol for a given company name in the context of a sentence.
    
    Args:
        company_name: Company name to search for
        sentence: Context sentence containing the company reference
        thresholds: Array of distance thresholds to try for matching
        
    Returns:
        Stock symbol if found, None otherwise
    """
    # Clean inputs
    sentence = clean_text(sentence)
    company_name = clean_text(company_name)
    company_words = company_name.lower().split()
    
    # 1. Direct symbol match check
    if company_name.upper() in symbol_to_name:
        symbol = company_name.upper()
        print(f"Matched symbol: {symbol}")
        return symbol
    
    # 2. Check for symbols in sentence that match company words
    sentence_words = sentence.split()
    potential_symbols = [
        word for word in sentence_words 
        if word in symbol_to_name and 
        all(c_word in symbol_to_name[word].lower() for c_word in company_words)
    ]
    
    # If there's only one potential symbol, return it
    if len(potential_symbols) == 1:
        symbol = potential_symbols[0]
        print(f"Matched symbol: {symbol}")
        return symbol
    
    # 3. If multiple potential symbols, perform semantic search
    if len(potential_symbols) > 1:
        # Generate context embeddings
        input_context = company_name + ' ' + sentence
        input_embedding = model.encode([input_context])
        
        # Prepare the sentence contexts and embeddings for the potential matches
        potential_contexts = [symbol_to_name[symbol] + ' ' + sentence for symbol in potential_symbols]
        potential_embeddings = model.encode(potential_contexts)
        
        # Calculate cosine similarities
        similarities = np.dot(input_embedding, potential_embeddings.T) / (
            np.linalg.norm(input_embedding) * np.linalg.norm(potential_embeddings, axis=1)
        )
        
        # Find the symbol with the highest similarity
        best_matching_symbol = potential_symbols[np.argmax(similarities)]
        print(f"Matched symbol: {best_matching_symbol}")
        return best_matching_symbol
    
    # 4. Context-based search using spaCy
    # Create context windows around the company_name in the sentence
    context_sentence = company_name + ' ' + sentence
    doc = nlp(context_sentence)
    num_words = min(len(company_name.split()), len(doc))
    
    # Extract context windows
    context_windows = []
    for i in range(num_words, len(doc)+1):
        window = doc[i-num_words:i]
        context_windows.append(window.text)
    
    # Encode the context windows
    context_sentence_embedding = model.encode(context_windows)
    
    # Filter companies that contain any words from company_name
    companies = [
        name for name in name_to_symbol 
        if any(word in name.lower() for word in company_words)
    ]
    
    # If no companies are found, return None
    if not companies:
        print(f"No companies found for {company_name}.")
        return None
    
    # Create a FAISS index for semantic search
    index = faiss.IndexFlatL2(context_sentence_embedding.shape[1])
    
    # Prepare embeddings for all candidate companies with the sentence context
    candidate_embeddings = [
        model.encode([candidate + ' ' + sentence]) for candidate in companies
    ]
    
    # Add the embeddings to the index
    index.add(np.array(candidate_embeddings).reshape(-1, context_sentence_embedding.shape[1]))
    
    # Perform a search
    D, I = index.search(context_sentence_embedding.astype('float32'), len(companies))
    
    # Try different thresholds to find a match
    for threshold in thresholds:
        if D[0][0] <= threshold:
            best_matching_company = companies[I[0][0]]
            best_matching_symbol = name_to_symbol[best_matching_company]
            print(f"Threshold: {threshold}, Matched symbol: {best_matching_symbol}")
            return best_matching_symbol
        else:
            print(f"Threshold: {threshold}, No match found.")
    
    print("No matching symbol found at any threshold")
    return None


# -------------------- Test Cases --------------------

if __name__ == "__main__":
    # Test cases for check_if_company_exist_in_tweet
    test_tweets = [
        "Is #India the new battleground for #streaming giants like $DIS and $NFLX?",
        "JPMorgan upgrades Taser maker: 'Pullback we were waiting for came sooner than expected'",
        "Meta is releasing a new VR headset next quarter.",
        "Family offices move to 'risk on' with plans to load up on stocks, private credit"
    ]
    
    # Test cases for symbol_per_company_name
    test_company_sentences = [
        ("Apple", "Apple is in the running for the best AI assistant, but can they pull ahead of the competition?"),
        ("Unity", "For the balance of 2023, we expect revenue to grow faster than the markets in which we compete."),
        ("$CEEK VR", "Join us tomorrow for an exclusive interview with the minds behind the innovative CEEK VR.")
    ]
    
    print("\n===== Testing check_if_company_exist_in_tweet =====")
    for tweet in test_tweets:
        print(f"\nTweet: {tweet}")
        check_if_company_exist_in_tweet(tweet)
    
    print("\n===== Testing symbol_per_company_name =====")
    for company, sentence in test_company_sentences:
        print(f"\nCompany: {company}")
        print(f"Sentence: {sentence}")
        symbol_per_company_name(company, sentence)