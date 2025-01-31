import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from tqdm import tqdm
import matplotlib.patches as mpatches

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english").to(device)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple", device=0 if device.type == "cuda" else -1)
nlp = spacy.load("en_core_web_sm")

# Function to extract entities
def extract_entities(text):
    doc = nlp(text)
    entities = {"People": [], "Organizations": [], "Crimes": [], "Locations": [], "Dates": []}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["People"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["Organizations"].append(ent.text)
        elif ent.label_ == "LAW" or "crime" in ent.text.lower():
            entities["Crimes"].append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            entities["Locations"].append(ent.text)
        elif ent.label_ in ["DATE", "TIME"]:
            entities["Dates"].append(ent.text)
    return entities

# Extract relationships
def extract_relationships(text, entities):
    doc = nlp(text)
    relationships = []
    for ent1 in entities:
        for ent2 in entities:
            if ent1 != ent2:
                rel = extract_relationship_from_text(text, ent1, ent2)
                if rel:
                    relationships.append((ent1, ent2, rel))
    return relationships

# Heuristic-based relationship extraction
def extract_relationship_from_text(text, ent1, ent2):
    rel_patterns = {
        "works with": ["works", "collaborates", "associates"],
        "stole from": ["stole", "robbed", "thief", "looted", "burgled"],
        "located in": ["located", "found", "situated in", "resides in"],
        "is a part of": ["part of", "member of", "included in", "belongs to"],
        "attacked": ["stabbed", "attacked", "injured", "assaulted", "hit", "shot"],
        "investigated by": ["investigated", "probed", "questioned by", "examined by"],
        "reported by": ["reported", "wrote about", "covered by", "published by"],
        "sued by": ["sued", "taken to court by", "charged by", "prosecuted by"],
        "arrested by": ["arrested", "detained by", "taken into custody by", "held by"],
        "related to": ["related", "connected to", "linked to", "associated with"]
    }
    for rel, verbs in rel_patterns.items():
        for verb in verbs:
            if verb in text.lower() and ent1.lower() in text.lower() and ent2.lower() in text.lower():
                return rel
    return None

# Visualize relationships
def visualize_relationships(entities, relationships):
    G = nx.Graph()
    G.add_nodes_from(entities)
    for ent1, ent2, label in relationships:
        G.add_edge(ent1, ent2, label=label)
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')
    plt.title("Entity Relationship Graph")
    plt.show()

# Process data
def process_data(file_path, output_file=None, visualize=False):
    data = pd.read_excel(file_path)
    tqdm.pandas(desc="Processing data")
    data["Entities"] = data["Text"].progress_apply(lambda x: list(set(sum(extract_entities(x).values(), []))))
    data["Relationships"] = data.apply(lambda row: extract_relationships(row["Text"], row["Entities"]), axis=1)
    
    if visualize:
        for _, row in data.iterrows():
            visualize_relationships(row["Entities"], row["Relationships"])
    
    if output_file:
        data.to_excel(output_file, index=False)
    return data

# User choice for output mode
def main():
    file_path = "Datasets/news_excerpts_parsed.xlsx"
    output_file = "processed_results.xlsx"
    
    print("Choose an option:")
    print("1. Generate Excel file")
    print("2. Visualize Graph")
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        process_data(file_path, output_file, visualize=False)
        print(f"Data saved to {output_file}")
    elif choice == "2":
        process_data(file_path, visualize=True)
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
