import pandas as pd
import spacy
import networkx as nx
import matplotlib.pyplot as plt

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract entities
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to extract relationships (subject-verb-object)
def extract_relationships(text):
    doc = nlp(text)
    relationships = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ("nsubj", "dobj"):  # Subject or object
                subject = [w.text for w in token.head.lefts if w.dep_ == "nsubj"]
                verb = token.head.text
                obj = [w.text for w in token.head.rights if w.dep_ == "dobj"]
                if subject and obj:
                    relationships.append((subject[0], verb, obj[0]))
    return relationships

# Load the Excel file
file_path = "Datasets/news_excerpts_parsed.xlsx"
df = pd.read_excel(file_path)

# Debugging: Print column names
print("Columns in the DataFrame:", df.columns)

# Strip leading/trailing spaces from column names (optional, for safety)
df.columns = df.columns.str.strip()

# Check if the 'Text' column exists
if 'Text' in df.columns:
    # Extract entities and relationships from the 'Text' column
    df["entities"] = df["Text"].apply(extract_entities)
    df["relationships"] = df["Text"].apply(extract_relationships)
    
    # Display the first few rows with extracted entities and relationships
    print(df[["Text", "entities", "relationships"]].head())
    
    # Create a graph to visualize relationships
    G = nx.Graph()

    # Add nodes (entities) and edges (relationships) to the graph
    for _, row in df.iterrows():
        entities = row["entities"]
        relationships = row["relationships"]
        
        # Add entities as nodes
        for entity in entities:
            G.add_node(entity[0], label=entity[1])
        
        # Add relationships as edges
        for rel in relationships:
            G.add_edge(rel[0], rel[2], label=rel[1])

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # Layout for consistent visualization
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", font_size=10, node_size=2000)
    
    # Add edge labels (verbs)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
    
    # Display the graph
    plt.title("Entity-Relationship Graph for News Excerpts")
    plt.show()

    # Save the results to a new Excel file
    output_file_path = "output_with_entities_and_relationships.xlsx"
    df.to_excel(output_file_path, index=False)
    print(f"Entity and relationship extraction complete! Results saved to '{output_file_path}'.")
else:
    print("Error: Column 'Text' not found. Available columns:", df.columns)