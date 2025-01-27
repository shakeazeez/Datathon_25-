import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from tqdm import tqdm  # For progress bar
import matplotlib.patches as mpatches

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pretrained BERT model and tokenizer for Named Entity Recognition (NER)
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple", device=0 if device.type == "cuda" else -1)

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

    # Limit entities to 75% of the total number found
    entity_limit = int(len(entities) * 0.75)
    entities = entities[:entity_limit]

    # Form relationships with meaningful labels based on text content
    relationships = []
    for i, ent1 in enumerate(entities):
        for ent2 in entities[i + 1:]:
            # Try to find relationships between entities by analyzing text for certain patterns (this is an example heuristic)
            relationship_label = extract_relationship_from_text(text, ent1, ent2)
            
            # If no relationship found, fall back to similarity
            if not relationship_label:
                relationship_label = f"Similarity: {compute_similarity(ent1, ent2):.2f}"

            relationships.append((ent1, ent2, relationship_label))
    
    return entities, relationships

# Function to extract a relationship based on text context (simple heuristic example)
def extract_relationship_from_text(text, ent1, ent2):
    # Simple heuristics to match verbs or actions between entities
    possible_relationships = [
        ("works with", ["works", "collaborates", "associates"]),
        ("stole from", ["stole", "robbed", "thief"]),
        ("located in", ["located", "found"]),
        ("is a part of", ["part of", "member of", "included in"]),
        ("stabbed", ["stabbed", "attacked", "injured"]),
        # Add more relationship patterns here as needed
    ]

    for relationship, verbs in possible_relationships:
        for verb in verbs:
            if verb in text.lower() and ent1.lower() in text.lower() and ent2.lower() in text.lower():
                return relationship
    
    return None

# Compute similarity between two entities (fallback when no relationship found)
def compute_similarity(ent1, ent2):
    encoded_ent1 = tokenizer.encode(ent1, return_tensors='pt').to(device)
    encoded_ent2 = tokenizer.encode(ent2, return_tensors='pt').to(device)
    min_dim = min(encoded_ent1.shape[1], encoded_ent2.shape[1])
    encoded_ent1 = encoded_ent1[:, :min_dim]
    encoded_ent2 = encoded_ent2[:, :min_dim]
    similarity = cosine_similarity(encoded_ent1.cpu().numpy(), encoded_ent2.cpu().numpy())[0][0]
    return similarity

# Visualize relationships as a network graph with labels on lines
def visualize_relationships(entities, relationships):
    G = nx.Graph()
    G.add_nodes_from(entities)
    
    # Add edges with relationship labels as edge attributes
    for ent1, ent2, label in relationships:
        G.add_edge(ent1, ent2, label=label)
    
    if len(entities) > 0:  # Proceed if there are any entities
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 10))
        
        # Draw nodes with rectangular shapes
        node_rectangles = {}
        for node, (x, y) in pos.items():
            node_rectangles[node] = [x - 0.05, y - 0.05, x + 0.05, y + 0.05]  # Adjust rectangle size

        # Draw the rectangular nodes
        for node, rect in node_rectangles.items():
            plt.gca().add_patch(mpatches.FancyBboxPatch((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1], 
                                                        boxstyle="round,pad=0.05", ec="black", fc="lightblue"))
            plt.text((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2, node, 
                     ha="center", va="center", fontsize=10)

        # Draw edges
        nx.draw_networkx_edges(G, pos, edgelist=relationships, edge_color='gray', width=1.0)
        
        # Draw edge labels (the relationship names)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')
        
        plt.title("Entity Relationship Graph")
        plt.axis('off')  # Hide axes
        plt.show()

# Process multiple files with batch processing
def process_in_batches(texts, batch_size=32):
    all_results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing in Batches"):
        batch = texts[i:i + batch_size]
        batch_results = ner_pipeline(batch)  # Process a batch
        all_results.extend(batch_results)  # Collect results
    return all_results

# Process multiple files with progress bar
def process_files(file_paths, expected_columns, output_file=None):
    all_data = []
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        data = load_and_validate_dataset(file_path, expected_columns)

        # Apply functions with a progress bar
        tqdm.pandas(desc=f"Processing {file_path}")
        data["Entities"], data["Relationships"] = zip(
            *data["Text"].progress_apply(extract_entities_and_relationships)
        )

        # Visualize relationships as a network graph after processing each file
        for entities, relationships in zip(data["Entities"], data["Relationships"]):
            visualize_relationships(entities, relationships)
        
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
