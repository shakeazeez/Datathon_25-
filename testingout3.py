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

# -------------------- Configuration --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use: {DEVICE}")

# Model configuration: Adjust MODEL_NAME to change speed/accuracy trade-offs
MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"  # Can be replaced with a faster/lighter model
ENTITY_LIMIT_FACTOR = 0.75  # Adjust to filter less meaningful entities (lower = stricter filtering)
BATCH_SIZE = 32  # Modify to process larger/smaller text batches for performance tuning
FILE_PATHS = ["Datasets/news_excerpts_parsed.xlsx", "Datasets/wikileaks_parsed.xlsx"]
OUTPUT_FILE = "combined_results.xlsx"  # Output file for processed results

# Initialize BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME).to(DEVICE)
nlp_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Initialize spaCy with neural model for better dependency parsing
nlp = spacy.load("en_core_web_trf")

# Relationship patterns
def init_relationship_matcher():
    matcher = Matcher(nlp.vocab)
    patterns = [
        [{"DEP": "nsubj"}, {"DEP": "prep"}, {"DEP": "pobj"}],
        [{"DEP": "nsubj"}, {"DEP": "dobj"}],
        [{"DEP": "nsubjpass"}, {"DEP": "prep"}],
    ]
    for pattern in patterns:
        matcher.add("RELATION", [pattern])
    return matcher

matcher = init_relationship_matcher()

# -------------------- Functions --------------------
def clean_entities(entities):
    """Removes duplicates and ensures entity names are meaningful."""
    clean_set = set()
    meaningful_entities = []
    for word, entity in entities:
        word = word.strip()
        if word not in clean_set and len(word) > 2 and '#' not in word:  # Filter short/irrelevant names
            clean_set.add(word)
            meaningful_entities.append((word, entity))
    return meaningful_entities

def extract_entities(text):
    """Extracts entities from text using the NER pipeline."""
    ner_results = nlp_pipeline(text)  # Extract named entities
    entities = [(result["word"], result["entity"]) for result in ner_results]
    return clean_entities(entities)  # Clean and deduplicate entities

def extract_relationships(text, entities):
    """Extracts relationships between entities in text."""
    doc = nlp(text)
    relationships = []

    for ent1, _ in entities:
        for ent2, _ in entities:
            if ent1 != ent2:  # Ensure relationships are not self-loops
                for sent in doc.sents:  # Process sentence by sentence
                    if ent1 in sent.text and ent2 in sent.text:
                        matches = matcher(sent)
                        for _, start, end in matches:
                            span = sent[start:end]
                            if ent1 in span.text and ent2 in span.text:
                                relationships.append((ent1, ent2, span.text))
                                break

    if not relationships:
        print(f"No relationships found in text: {text}")
    else:
        print(f"Extracted relationships from text: {relationships}")

    return relationships

def process_files():
    """Processes input files to extract entities and relationships."""
    all_data = []

    for file_path in FILE_PATHS:
        df = pd.read_excel(file_path)
        texts = df["Text"].tolist()  # Assuming a "Text" column
        dataset = Dataset.from_dict({"text": texts})

        # Process in batches for efficiency
        for batch in tqdm(dataset.to_dict()["text"], desc=f"Processing {file_path}"):
            entities = extract_entities(batch)  # Extract entities
            print(f"Extracted entities: {entities}")
            relationships = extract_relationships(batch, entities)  # Extract relationships
            all_data.append({"Text": batch, "Entities": entities, "Relationships": relationships})

    # Save results to an output file
    processed_data = pd.DataFrame(all_data)
    processed_data.to_excel(OUTPUT_FILE, index=False)
    print(f"Data saved to {OUTPUT_FILE}")

def visualize_relationships(entities, relationships, title="Entity Relationship Graph"):
    """Visualizes relationships using NetworkX."""
    G = nx.DiGraph()  # Use Directed Graph for better visual clarity
    G.add_nodes_from([ent[0] for ent in entities])

    # Add edges with labels
    for ent1, ent2, label in relationships:
        if ent1 in G.nodes and ent2 in G.nodes:
            G.add_edge(ent1, ent2, label=label)

    # Visualization layout configuration
    pos = nx.spring_layout(G, k=0.5)  # Adjust 'k' to modify node spacing
    plt.figure(figsize=(15, 12))
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="lightgreen", alpha=0.9)  # Customize node size/color
    nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.7, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="blue", font_size=9)

    # Finalize and display the plot
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def load_and_visualize():
    """Loads processed data and visualizes relationships."""
    try:
        # Load processed results from the output file
        df = pd.read_excel(OUTPUT_FILE)
        df["Entities"] = df["Entities"].apply(ast.literal_eval)  # Parse stringified lists
        df["Relationships"] = df["Relationships"].apply(ast.literal_eval)

        # Visualize each text's relationships
        for _, row in df.iterrows():
            visualize_relationships(row["Entities"], row["Relationships"], title="Entity Relationship Graph")

    except FileNotFoundError:
        print(f"Error: {OUTPUT_FILE} not found. Run processing first.")

# -------------------- Execution --------------------
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
