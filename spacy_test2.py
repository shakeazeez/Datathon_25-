import spacy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from spacy import displacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load dataset
file_path = "Datasets/news_excerpts_parsed.xlsx"  # Change this if needed
df = pd.read_excel(file_path)

# Function to extract named entities
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Function to extract meaningful relationships
def extract_relationships(text):
    doc = nlp(text)
    relationships = []

    # Extract subject-verb-object triplets
    for token in doc:
        # Identify subject and object
        subject = None
        object_ = None

        if token.dep_ in ["ROOT", "xcomp", "ccomp"]:  # Focus on main verbs
            verb = token.text
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:  # Find subject
                    subject = child.text
                if child.dep_ in ["dobj", "attr", "prep", "pobj"]:  # Find object
                    object_ = child.text

            # Ensure we have a valid triplet (subject, verb, object)
            if subject and object_:
                relationships.append((subject, verb, object_))

    return relationships

# Process first 5 rows
for idx, text in enumerate(df["Text"][:5]):
    print(f"\n--- Processing Text {idx + 1} ---\n")

    # Extract entities and relationships
    entities = extract_entities(text)
    relationships = extract_relationships(text)

    # Display extracted entities
    print("Extracted Entities:")
    for entity in entities:
        print(entity)

    # Display extracted relationships
    print("\nExtracted Relationships:")
    if relationships:
        for relation in relationships:
            print(relation)
    else:
        print("No significant relationships found.")

    # Visualize Named Entities
    doc = nlp(text)
    displacy.render(doc, style="ent", jupyter=True)

    # Create a network graph if relationships exist
    if relationships:
        G = nx.DiGraph()
        for subject, action, object_ in relationships:
            G.add_edge(subject, object_, label=action)

        # Draw graph
        plt.figure(figsize=(8, 5))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
        edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        plt.title(f"Entity Relationship Graph - Text {idx + 1}")
        plt.show()
