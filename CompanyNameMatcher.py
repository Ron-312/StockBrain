import spacy

class CompanyNameMatcher:
    def __init__(self):
        # Load the medium-sized spaCy model for English
        self.nlp = spacy.load("en_core_web_md")
    
    def is_same_company(self, name1, name2, similarity_threshold=0.8):
        """
        Determine if two company names refer to the same entity based on NLP similarity.
        
        :param name1: The first company name.
        :param name2: The second company name.
        :param similarity_threshold: The threshold for considering names to refer to the same company.
        :return: True if the names are likely the same company, False otherwise.
        """
        # Process the company names to get their vector representations
        doc1 = self.nlp(name1)
        doc2 = self.nlp(name2)
        
        # Calculate similarity - returns a number between 0 and 1
        similarity = doc1.similarity(doc2)
        
        # Check against the threshold
        return similarity >= similarity_threshold
