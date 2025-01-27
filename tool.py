import spacy
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract specific entities
def extract_entities(text):
    doc = nlp(text)
    entities = {
        "People": [ent.text for ent in doc.ents if ent.label_ in ["PERSON"]],
        "Organizations": [ent.text for ent in doc.ents if ent.label_ in ["ORG"]],
        "Crimes": [ent.text for ent in doc.ents if "crime" in ent.text.lower() or ent.label_ == "LAW"],
        "Locations": [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]],
        "Dates": [ent.text for ent in doc.ents if ent.label_ in ["DATE", "TIME"]],
    }
    return entities

# Function to extract relationships
def extract_relationships(text):
    doc = nlp(text)
    relationships = []

    for token in doc:
        # Check if the token is a subject and its head is a verb
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            # Ensure the subject is meaningful
            if token.pos_ not in ["PRON", "DET", "ADP", "CCONJ"]:  # Exclude pronouns, determiners, prepositions, and connectors
                subject = token.text
                verb = token.head.text

                # Find objects of the verb, excluding prepositions and connectors
                objects = [
                    child.text
                    for child in token.head.children
                    if child.dep_ in ["dobj", "pobj"] and child.pos_ not in ["ADP", "CCONJ"]
                ]

                for obj in objects:
                    relationships.append({"Subject": subject, "Relationship": verb, "Object": obj})

    return relationships


# Data validation function
def validate_data(df, expected_columns):
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    if df.isnull().any().any():
        print("Warning: Missing values detected.")
    return df.drop_duplicates().dropna()

# Updated function to process multiple files
def process_multiple_files(file_paths, columns, output_file=None):
    """
    Processes multiple Excel files to extract entities and relationships.
    
    Args:
        file_paths (list): List of file paths to analyze.
        columns (list): List of required columns in the datasets.
        output_file (str, optional): Path to save the combined output Excel file.
    
    Returns:
        pd.DataFrame: Combined DataFrame with extracted entities and relationships from all files.
    """
    all_data = []
    
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        
        # Load dataset
        data = pd.read_excel(file_path)
        
        # Validate dataset
        data = validate_data(data, columns)
        
        # Extract entities and relationships
        data["entities"] = data["Text"].apply(extract_entities)
        data["relationships"] = data["Text"].apply(extract_relationships)
        
        # Keep track of data from all files
        all_data.append(data)
    
    # Combine data from all files
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Save results if output file is provided
    if output_file:
        combined_data.to_excel(output_file, index=False)
    
    return combined_data

# List of files to analyze
file_paths = ["Datasets/news_excerpts_parsed.xlsx", "Datasets/wikileaks_parsed.xlsx"]

# Required columns in the files
columns = ["Text"]

# Process the files and save the results
output_file = "combined_results.xlsx"
processed_data = process_multiple_files(file_paths, columns, output_file=output_file)

# Print sample results
print(processed_data.head())
