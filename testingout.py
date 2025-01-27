import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import spacy

# Load pretrained BERT model and tokenizer for Named Entity Recognition (NER)
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract specific entities using spaCy
def extract_spacy_entities(text):
    doc = nlp(text)
    entities = {
        "People": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
        "Organizations": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
        "Crimes": [ent.text for ent in doc.ents if "crime" in ent.text.lower() or ent.label_ == "LAW"],
        "Locations": [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]],
        "Dates": [ent.text for ent in doc.ents if ent.label_ in ["DATE", "TIME"]],
    }
    return entities

# Load and validate dataset
def load_and_validate_dataset(file_path, expected_columns):
    data = pd.read_excel(file_path)
    for col in expected_columns:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")
    if data.isnull().any().any():
        print(f"Warning: Missing values in {file_path}")
    return data.drop_duplicates().dropna()

# Function to extract entities and relationships from a single text
def extract_entities_and_relationships(text):
    # Extract entities using the NER pipeline
    ner_results = ner_pipeline(text)
    entities = [res['word'] for res in ner_results if res['entity_group'] in ['PER', 'ORG', 'LOC']]

    # Extract additional entities using spaCy
    spacy_entities = extract_spacy_entities(text)
    for entity_list in spacy_entities.values():
        entities.extend(entity_list)

    # Remove duplicates from entities
    entities = list(set(entities))

    # Form relationships (placeholder logic)
    relationships = []
    for i, ent1 in enumerate(entities):
        for ent2 in entities[i + 1:]:
            encoded_ent1 = tokenizer.encode(ent1, return_tensors='pt')
            encoded_ent2 = tokenizer.encode(ent2, return_tensors='pt')
            min_dim = min(encoded_ent1.shape[1], encoded_ent2.shape[1])
            encoded_ent1 = encoded_ent1[:, :min_dim]
            encoded_ent2 = encoded_ent2[:, :min_dim]
            similarity = cosine_similarity(encoded_ent1.numpy(), encoded_ent2.numpy())
            relationships.append((ent1, ent2, similarity[0][0]))
    
    return entities, relationships

# Visualize relationships as a network graph
def visualize_relationships(entities, relationships):
    G = nx.Graph()
    G.add_nodes_from(entities)
    for ent1, ent2, similarity in relationships:
        if similarity > 0.5:
            G.add_edge(ent1, ent2, weight=similarity)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
    plt.title("Entity Relationship Graph")
    plt.show()

# Process multiple files
def process_files(file_paths, expected_columns, output_file=None):
    all_data = []
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        data = load_and_validate_dataset(file_path, expected_columns)
        data["Entities"], data["Relationships"] = zip(*data["Text"].apply(extract_entities_and_relationships))
        all_data.append(data)
    combined_data = pd.concat(all_data, ignore_index=True)
    if output_file:
        combined_data.to_excel(output_file, index=False)
    return combined_data

# File paths and expected columns
file_paths = ["Datasets/news_excerpts_parsed.xlsx", "Datasets/wikileaks_parsed.xlsx"]
expected_columns = ["Text"]

# Process and save results
output_file = "combined_results.xlsx"
processed_data = process_files(file_paths, expected_columns, output_file=output_file)

# Print sample results
print(processed_data.head())
