import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load pretrained BERT model and tokenizer for Named Entity Recognition (NER)
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")

# Load dataset from CSV file
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    if 'Text' not in data.columns:
        raise ValueError("The dataset must contain a 'Text' column.")
    return data

# Function to extract entities and relationships
def extract_entities_and_relationships(text):
    # Step 1: Extract entities using NER pipeline
    ner_results = ner_pipeline(text)

    # Step 2: Form entity pairs and compute relationships
    entities = [res['word'] for res in ner_results if res['entity_group'] in ['PER', 'ORG', 'LOC']]
    relationships = []
    for i, ent1 in enumerate(entities):
        for ent2 in entities[i + 1:]:
            # Encode entities as token embeddings
            encoded_ent1 = tokenizer.encode(ent1, return_tensors='pt')
            encoded_ent2 = tokenizer.encode(ent2, return_tensors='pt')

            # Debug: Print shapes
            print(f"Encoded entity 1 shape: {encoded_ent1.shape}")
            print(f"Encoded entity 2 shape: {encoded_ent2.shape}")

            # Ensure compatibility by aligning dimensions
            min_dim = min(encoded_ent1.shape[1], encoded_ent2.shape[1])
            encoded_ent1 = encoded_ent1[:, :min_dim]
            encoded_ent2 = encoded_ent2[:, :min_dim]

            # Compute similarity
            similarity = cosine_similarity(encoded_ent1.numpy(), encoded_ent2.numpy())
            relationships.append((ent1, ent2, similarity[0][0]))  # Convert tensor similarity to scalar

    return entities, relationships

# Visualize relationships as a network graph
def visualize_relationships(entities, relationships):
    G = nx.Graph()

    # Add nodes for entities
    G.add_nodes_from(entities)

    # Add edges for relationships
    for ent1, ent2, similarity in relationships:
        if similarity > 0.5:  # Threshold to filter meaningful relationships
            G.add_edge(ent1, ent2, weight=similarity)

    # Draw the graph
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
    plt.title("Entity Relationship Graph")
    plt.show()

# Example usage
file_path = "Datasets/news_excerpts_parsed.csv"  # Replace with your CSV file path
data = load_dataset(file_path)

# Process first sample for demonstration
sample_text = data['Text'][0]
entities, relationships = extract_entities_and_relationships(sample_text)

# Display extracted entities and relationships
print("Entities:", entities)
print("Relationships:", relationships)

# Visualize relationships
visualize_relationships(entities, relationships)
