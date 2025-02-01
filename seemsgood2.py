from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from tqdm import tqdm

# Load pretrained BERT model and tokenizer for Named Entity Recognition (NER)
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")

# Load SpaCy for dependency parsing
nlp = spacy.load("en_core_web_sm")

# Load dataset
def load_dataset(file_path):
    data = pd.read_excel(file_path)
    if 'Text' not in data.columns:
        raise ValueError("The dataset must contain a 'Text' column.")
    return data

# Merge tokens correctly
def merge_tokens(text):
    tokens = tokenizer.tokenize(text)
    merged_text = "".join([t.replace("##", "") if t.startswith("##") else " " + t for t in tokens]).strip()
    return merged_text

# Extract named entities using BERT
def extract_named_entities(text):
    entities = ner_pipeline(text)
    entity_map = {ent['word']: ent['entity_group'] for ent in entities}
    return entity_map

# Improved Relationship Extraction
def extract_relationships(text):
    merged_text = merge_tokens(text)
    doc = nlp(merged_text)
    named_entities = extract_named_entities(merged_text)  # Get named entities
    relationships = []

    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            subject = token.text
            verb = token.head.text
            objects = [child.text for child in token.head.children if child.dep_ in ["dobj", "pobj", "attr"]]

            # Check for additional context like negation, adverbs, and adjectives
            negation = [child.text for child in token.head.children if child.dep_ == "neg"]
            adverbs = [child.text for child in token.head.children if child.dep_ == "advmod"]
            modifiers = [child.text for child in token.children if child.dep_ == "amod"]

            verb_phrase = " ".join(negation + adverbs + [verb])  # Combine negation and adverbs with verb
            for obj in objects:
                obj_phrase = " ".join(modifiers + [obj])  # Add adjectives to object

                # Ensure subject and object are properly named entities
                subject = named_entities.get(subject, subject)
                obj_phrase = named_entities.get(obj_phrase, obj_phrase)

                relationships.append({"Subject": subject, "Action": verb_phrase, "Object": obj_phrase})

    return relationships

# Visualization function
def visualize_relationships(relationships):
    G = nx.DiGraph()
    for rel in relationships:
        G.add_edge(rel["Subject"], rel["Object"], label=rel["Action"])
    
    pos = nx.spring_layout(G, k=3.0)
    plt.figure(figsize=(12, 8), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    nx.draw(
        G, pos, with_labels=True, node_color="lightblue", edge_color="black",
        node_size=3000, font_size=10, font_color="black", arrows=True,
        edgecolors="black", linewidths=1.5, width=2, alpha=0.8
    )
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black", font_size=12)
    
    plt.title("Entity Relationship Graph with Actions", color="black", fontsize=14)
    plt.show()

# Function to categorize relationships into crime categories
def map_to_crime_categories(relationships):
    crime_categories = {
        "Murder": ["murder", "kill", "homicide", "manslaughter"],
        "Thievery": ["steal", "theft", "rob", "burglary", "shoplift"],
        "Fraud": ["fraud", "scam", "deceive", "embezzle", "forge"],
        "Assault": ["assault", "attack", "violate", "batter"],
        "Kidnapping": ["kidnap", "abduct", "hostage"],
        "Bribery": ["bribe", "corrupt", "kickback"],
        "Cybercrime": ["hack", "cyber", "phish", "malware"],
    }
    
    categorized_data = []
    for rel in relationships:
        action = f"{rel['Action']} {rel['Object']}".lower()
        action_lemmas = [token.lemma_ for token in nlp(action)]
        for category, keywords in crime_categories.items():
            if any(keyword in action_lemmas for keyword in keywords):
                categorized_data.append({"Category": category, "Action": action})
                break
    return categorized_data

# Visualizing crime categories
def analyze_and_visualize_crimes(categorized_data):
    crime_counts = pd.DataFrame(categorized_data).value_counts("Category").reset_index()
    crime_counts.columns = ["Category", "Frequency"]

    print("Processing crime categories...")
    for _ in tqdm(range(10), desc="Generating visualization"):
        pass  

    fig = px.bar(crime_counts, x="Category", y="Frequency", title="Crime Categories", 
                 labels={"Category": "Crime Category", "Frequency": "Frequency"}, text_auto=True)
    fig.update_layout(xaxis_tickangle=45)
    fig.show()

# Main function
def main():
    file_path = "Datasets/news_excerpts_parsed.xlsx"
    data = load_dataset(file_path)
    
    while True:
        print("\nChoose an option:")
        print("1. Extract and save Relationships to File")
        print("2. Visualize Relationships")
        print("3. Visualize Crime Categories")
        print("4. Tokenize Text")
        print("Q. Quit")
        choice = input("Enter choice: ").strip().lower()
        
        if choice == "1":
            all_relationships = []
            for text in tqdm(data['Text'], desc="Processing relationships"):
                all_relationships.extend(extract_relationships(text))
            pd.DataFrame(all_relationships).to_excel("relationships.xlsx", index=False)
            print("Relationships saved to relationships.xlsx")
        
        elif choice == "2":
            if data.empty:
                print("No data available for visualization.")
            else:
                for text in data['Text']:
                    relationships = extract_relationships(text)
                    visualize_relationships(relationships)

        elif choice == "3":
            all_relationships = []
            for text in tqdm(data['Text'], desc="Processing relationships"):
                all_relationships.extend(extract_relationships(text))
            categorized_data = map_to_crime_categories(all_relationships)
            analyze_and_visualize_crimes(categorized_data)

        elif choice == "q":
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()
