import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load pretrained BERT model and tokenizer for Named Entity Recognition (NER)
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")

# Load SpaCy for dependency parsing
nlp = spacy.load("en_core_web_sm")

# Load dataset from CSV file
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    if 'Text' not in data.columns:
        raise ValueError("The dataset must contain a 'Text' column.")
    return data


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


# Visualize directional relationships as a directed graph
# ... Previous code for extracting entities and relationships ...

# Visualize directional relationships as a directed graph
def visualize_relationships(relationships):
    G = nx.DiGraph()



    # Add edges for relationships with labels
    for rel in relationships:
        print(f"Adding edge: {rel['Subject']} --{rel['Relationship']}--> {rel['Object']}")  # Debug
        G.add_edge(rel["Subject"], rel["Object"], label=rel["Relationship"])

    # Draw the graph with relationship labels
    pos = nx.spring_layout(G)  # Layout for graph
    plt.figure(figsize=(12, 8))
    nx.draw(
        G, pos, with_labels=True, node_color="lightblue", edge_color="gray",
        node_size=2000, font_size=10, arrowsize=20
    )

    # Add edge labels (relationship words)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=10)
    
    plt.title("Entity Relationship Graph with Actions")
    plt.show()

# Example usage
file_path = "Datasets/news_excerpts_parsed.csv"  # Replace with your CSV file path
data = load_dataset(file_path)

# Process the first sample for demonstration
for data_thing in data['Text']:
    print(data_thing)
    relationships = extract_relationships(data_thing)
    print("Relationships:", relationships)
    visualize_relationships(relationships)

print("Relationships:", relationships)

# Visualize relationships
visualize_relationships(relationships)

