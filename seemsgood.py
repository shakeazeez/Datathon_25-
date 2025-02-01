import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Load pretrained BERT model and tokenizer for Named Entity Recognition (NER)
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")

# Load SpaCy for dependency parsing
nlp = spacy.load("en_core_web_sm")

def load_dataset(file_path):
    data = pd.read_csv(file_path)
    if 'Text' not in data.columns:
        raise ValueError("The dataset must contain a 'Text' column.")
    return data

# Function to merge broken-up tokens
def merge_tokens(text):
    tokens = tokenizer.tokenize(text)
    merged_text = "".join([t.replace("##", "") if t.startswith("##") else " " + t for t in tokens]).strip()
    return merged_text

# Improved relationship extraction function
def extract_relationships(text):
    merged_text = merge_tokens(text)
    doc = nlp(merged_text)
    relationships = []

    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            subject = token.text
            verb = token.head.text
            
            # Identify objects and indirect objects
            objects = [child.text for child in token.head.children if child.dep_ in ["dobj", "pobj", "attr"]]
            
            # Identify additional nested relationships
            for obj in objects:
                relationships.append({"Subject": subject, "Relationship": verb, "Object": obj})
                for child in doc:
                    if child.head == token.head and child.dep_ in ["prep", "acl"]:
                        relationships.append({"Subject": obj, "Relationship": child.text, "Object": child.head.text})
    return relationships

# def extract_relationships(text):
#     merged_text = merge_tokens(text)
#     doc = nlp(merged_text)
#     relationships = []

#     # Extract named entities using BERT NER model
#     ner_results = ner_pipeline(merged_text)
#     named_entities = {ent['word']: ent['entity_group'] for ent in ner_results}

#     for token in doc:
#         # Only consider noun subjects
#         if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
#             subject = token.text
#             verb = token.head.text
            
#             # Identify objects (ensure they are nouns)
#             objects = [child.text for child in token.head.children if child.dep_ in ["dobj", "pobj", "attr"] and child.pos_ in ["NOUN", "PROPN"]]

#             # Ensure subject is a noun or named entity (not a verb)
#             if subject in named_entities or token.pos_ in ["NOUN", "PROPN"]:
#                 for obj in objects:
#                     if obj in named_entities or obj in ["NOUN", "PROPN"]:
#                         relationships.append({"Subject": subject, "Relationship": verb, "Object": obj})

#                         # Identify prepositional relationships
#                         for child in doc:
#                             if child.head == token.head and child.dep_ in ["prep", "acl"]:
#                                 relationships.append({"Subject": obj, "Relationship": child.text, "Object": child.head.text})

#     return relationships

# Improved visualization function
def visualize_relationships(relationships):
    G = nx.DiGraph()
    for rel in relationships:
        G.add_edge(rel["Subject"], rel["Object"], label=rel["Relationship"])
    
    pos = nx.spring_layout(G, k=2.0)  # Increased spacing
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10, arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=10)
    plt.title("Entity Relationship Graph with Actions")
    plt.show()

# Tokenization function with token type visualization
def tokenize_text(text):
    doc = nlp(text)
    tokens_info = [(token.text, token.pos_) for token in doc]
    print("Tokenized Text:")
    for token, pos in tokens_info:
        print(f"{token}: {pos}",end = ", ")

# Main menu-driven function
def main():
    file_path = "Datasets/news_excerpts_parsed.csv"  # Ensure it's a CSV file
    data = load_dataset(file_path)
    
    while True:
        print("\nChoose an option:")
        print("1. Extract and save relationships to file")
        print("2. Visualize relationships")
        print("3. Tokenize text")
        print("Q. Quit")
        choice = input("Enter choice: ").strip().lower()
        
        if choice == "1":
            all_relationships = []
            for text in tqdm(data['Text'], desc="Processing relationships"):
                all_relationships.extend(extract_relationships(text))
            pd.DataFrame(all_relationships).to_csv("relationships.csv", index=False)
            print("Relationships saved to relationships.csv")
        
        elif choice == "2":
            if data.empty:
                print("No data available for visualization.")
            else:
                for datas in data['Text']:
                    relationships = extract_relationships(datas)
                    visualize_relationships(relationships)
            
        elif choice == "3":
            count = 0
            while True:
                sample_text = data['Text'][count] if not data.empty else "No data available"
                count += 1
                tokenize_text(sample_text)
                print("\n\nEnter Y to continue seeing tokens, N to exit!")
                out = input("Enter choice: ").strip().lower()
                if out == 'n':
                    break
        
        elif choice == "q":
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()
