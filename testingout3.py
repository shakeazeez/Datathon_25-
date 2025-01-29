import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import Dataset
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import ast

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use: {DEVICE}")

MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"
ENTITY_LIMIT_FACTOR = 0.75
BATCH_SIZE = 32
FILE_PATHS = ["Datasets/news_excerpts_parsed.xlsx", "Datasets/wikileaks_parsed.xlsx"]
OUTPUT_FILE = "combined_results.xlsx"

# Initialize BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME).to(DEVICE)
nlp_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Initialize spaCy with neural model for better dependency parsing
nlp = spacy.load("en_core_web_trf")

# Enhanced Relationship patterns
def init_relationship_matcher():
    matcher = Matcher(nlp.vocab)
    patterns = [
        [{"DEP": "nsubj"}, {"DEP": "prep"}, {"DEP": "pobj"}],
        [{"DEP": "nsubj"}, {"DEP": "dobj"}],
        [{"DEP": "nsubjpass"}, {"DEP": "prep"}],
        [{"DEP": "num"}, {"DEP": "quantmod"}],  # Fixed numerical relationships
        [{"DEP": "appos"}]  # Fixed appositive relationships
    ]
    for pattern in patterns:
        matcher.add("RELATION", [pattern])
    return matcher

matcher = init_relationship_matcher()

def extract_entities(text):
    """Extracts entities from text using the NER pipeline."""
    ner_results = nlp_pipeline(text)
    entities = list(set([(result["word"], result["entity"]) for result in ner_results]))
    return entities

def extract_relationships(text, entities):
    """Extracts relationships between entities in text."""
    doc = nlp(text)
    relationships = []

    for ent1, _ in entities:
        for ent2, _ in entities:
            if ent1 != ent2:
                for sent in doc.sents:
                    if ent1 in sent.text and ent2 in sent.text:
                        matches = matcher(sent)
                        for _, start, end in matches:
                            span = sent[start:end]
                            if ent1 in span.text and ent2 in span.text:
                                relationships.append((ent1, ent2, span.text))
                                break

    for sent in doc.sents:
        for token in sent:
            if token.lemma_ in ["be", "have", "say"]:
                subjects = [child.text for child in token.head.children if child.dep_ in ["nsubj", "nsubjpass"]]
                objects = [child.text for child in token.head.children if child.dep_ in ["dobj", "pobj"]]
                for subj in subjects:
                    for obj in objects:
                        relationships.append((subj, obj, f"{token.lemma_} {token.head.text}"))

    return relationships

def process_files():
    """Processes input files to extract entities and relationships."""
    all_data = []

    for file_path in FILE_PATHS:
        df = pd.read_excel(file_path)
        texts = df["Text"].dropna().tolist()  # Fixed potential missing values
        dataset = Dataset.from_dict({"text": texts})

        for batch in tqdm(dataset["text"], desc=f"Processing {file_path}"):
            entities = extract_entities(batch)
            relationships = extract_relationships(batch, entities)
            all_data.append({"Text": batch, "Entities": entities, "Relationships": relationships})

    processed_data = pd.DataFrame(all_data)
    processed_data.to_excel(OUTPUT_FILE, index=False)
    print(f"Data saved to {OUTPUT_FILE}")

def visualize_relationships(entities, relationships, title="Entity Relationship Graph"):
    """Visualizes relationships using NetworkX with enhanced styling."""
    G = nx.Graph()
    G.add_nodes_from([e[0] for e in entities])
    meaningful_rels = [(u, v, l) for u, v, l in relationships if l]
    
    if not meaningful_rels:
        print("No meaningful relationships found to visualize.")
        return

    G.add_edges_from([(u, v, {"label": l}) for u, v, l in meaningful_rels])
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G)

    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="#4ECDC4", edge_color="#C4ECD4")
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.show()

def load_and_visualize():
    """Loads processed data and visualizes relationships."""
    try:
        df = pd.read_excel(OUTPUT_FILE)
        df["Entities"] = df["Entities"].apply(ast.literal_eval)
        df["Relationships"] = df["Relationships"].apply(ast.literal_eval)

        combined_G = nx.Graph()
        for _, row in df.iterrows():
            entities = row["Entities"]
            relationships = row["Relationships"]
            visualize_relationships(entities, relationships, title="Individual Document Relationships")

            combined_G.add_nodes_from([e[0] for e in entities])
            combined_G.add_edges_from([(rel[0], rel[1], {"label": rel[2]}) for rel in relationships])

        visualize_relationships(
            list(combined_G.nodes),
            [(u, v, d["label"]) for u, v, d in combined_G.edges(data=True)],
            title="Combined Relationship Graph"
        )
    except FileNotFoundError:
        print(f"Error: {OUTPUT_FILE} not found. Run processing first.")

if __name__ == "__main__":
    print("Select an option:")
    print("1: Process data files")
    print("2: Visualize relationships from saved data")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        print("Starting data processing...")
        process_files()
    elif choice == "2":
        print("Loading and visualizing data...")
        load_and_visualize()
    else:
        print("Invalid choice. Exiting.")
