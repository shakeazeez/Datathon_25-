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

def load_dataset(file_path):
    data = pd.read_excel(file_path)
    if 'Text' not in data.columns:
        raise ValueError("The dataset must contain a 'Text' column.")
    return data

def merge_tokens(text):
    tokens = tokenizer.tokenize(text)
    merged_text = "".join([t.replace("##", "") if t.startswith("##") else " " + t for t in tokens]).strip()
    return merged_text

def extract_relationships(text):
    merged_text = merge_tokens(text)
    doc = nlp(merged_text)
    relationships = []

    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            subject = token.text
            verb = token.head.text
            objects = [child.text for child in token.head.children if child.dep_ in ["dobj", "pobj", "attr"]]
            
            for obj in objects:
                relationships.append({"Subject": subject, "Relationship": verb, "Object": obj})
    return relationships



def visualize_relationships(relationships):
    # Create a directed graph
    G = nx.DiGraph()
    for rel in relationships:
        G.add_edge(rel["Subject"], rel["Object"], label=rel["Relationship"])
    
    # Define the layout of the graph for better spacing of nodes
    pos = nx.spring_layout(G, k=3.0)
    
    # Set up the figure with a white background
    plt.figure(figsize=(12, 8), facecolor='white')  # Set entire figure background to white
    ax = plt.gca()
    ax.set_facecolor('white')  # Ensure the axes background is also white
    
    # Draw the nodes and edges with the most visible colors
    nx.draw(
        G, pos, with_labels=True, 
        node_color="lightblue",  # Set nodes to light blue for visibility
        edge_color="black",  # Set edges to black for contrast against white background
        node_size=3000,  # Adjust node size
        font_size=10,  # Adjust label font size
        font_color="black",  # Set font color to black for contrast
        arrows=True,  # Ensure arrows are drawn for directed edges
        edgecolors="black",  # Add black border around nodes for better clarity
        linewidths=1.5,  # Set border width of nodes
        width=2,  # Set edge width for better visibility
        alpha=0.8  # Adjust transparency for clarity
    )
    
    # Draw edge labels to show relationships
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, 
        font_color="black",  # Set edge label color to black
        font_size=12  # Adjust font size
    )
    
    # Set title with black color for visibility
    plt.title("Entity Relationship Graph with Actions", color="black", fontsize=14)
    plt.show()


def map_to_crime_categories(relationships):
    crime_categories = {
    "Murder": ["murder", "kill", "homicide", "manslaughter"],
    "Thievery": ["steal", "theft", "rob", "burglary", "shoplift"],
    "Fraud": ["fraud", "scam", "deceive", "embezzle", "forge"],
    "Assault": ["assault", "attack", "violate", "batter"],
    "Kidnapping": ["kidnap", "abduct", "hostage"],
    "Arson": ["arson", "burn", "fire"],
    "Drug Trafficking": ["drug", "traffic", "narcotic"],
    "Cybercrime": ["hack", "cyber", "phish", "malware"],
    "Terrorism": ["terrorize", "bomb", "extremism", "radicalize"],
    "Bribery": ["bribe", "corrupt", "kickback"],
    "Sexual Assault": ["rape", "molest", "harass", "assault"],
    "Extortion": ["extort", "blackmail", "coerce"],
    "Money Laundering": ["launder", "money laundering", "clean money"],
    "Human Trafficking": ["traffick", "smuggle", "exploit"],
    "Vandalism": ["vandalize", "graffiti", "damage"],
    "Weapons Offense": ["firearm", "weapon", "gun", "knife"],
    "Smuggling": ["smuggle", "contraband", "illicit trade"],
    "Public Disorder": ["riot", "disturb", "unrest"],
    "Counterfeiting": ["counterfeit", "fake", "forge"],
    "Poaching": ["poach", "wildlife crime", "hunt illegally"],
    "Tax Evasion": ["evade", "tax fraud", "evasion"],
    "Perjury": ["perjure", "lie under oath", "false testimony"],
    "Espionage": ["spy", "espionage", "leak intelligence"],
    "Identity Theft": ["steal identity", "impersonate", "identity fraud"],
    "Illegal Immigration": ["immigrate illegally", "border cross", "undocumented"],
    "Domestic Violence": ["domestic abuse", "intimate partner violence"]
    }
    categorized_data = []
    for rel in relationships:
        action = f"{rel['Relationship']} {rel['Object']}".lower()
        action_lemmas = [token.lemma_ for token in nlp(action)]
        for category, keywords in crime_categories.items():
            if any(keyword in action_lemmas for keyword in keywords):
                categorized_data.append({"Category": category, "Action": action})
                break
    return categorized_data

def analyze_and_visualize_crimes(categorized_data):
    crime_counts = pd.DataFrame(categorized_data).value_counts("Category").reset_index()
    crime_counts.columns = ["Category", "Frequency"]
    
    print("Processing crime categories...")
    for _ in tqdm(range(10), desc="Generating visualization"):
        pass  # Simulate loading
    
    fig = px.bar(crime_counts, x="Category", y="Frequency", 
                 title="Crime Categories", 
                 labels={"Category": "Crime Category", "Frequency": "Frequency"},
                 text_auto=True)
    fig.update_layout(xaxis_tickangle=45)
    fig.show()

# Tokenization function with token type visualization
def tokenize_text(text):
    text2 = merge_tokens(text)
    doc = nlp(text2)
    tokens_info = [(token.text, token.pos_) for token in doc]
    print("\n\nTokenized Text:", end = ' ')
    for token, pos in tokens_info:
        print(f"{token}: {pos}",end = ", ")
    
    print("\n\nNormal Text:", end = ' ')
    for token, pos in tokens_info:
        print(f"{token}",end = " ")






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
                for datas in data['Text']:
                    relationships = extract_relationships(datas)
                    visualize_relationships(relationships)

        elif choice == "3":
            all_relationships = []
            for text in tqdm(data['Text'], desc="Processing relationships"):
                all_relationships.extend(extract_relationships(text))
            categorized_data = map_to_crime_categories(all_relationships)
            analyze_and_visualize_crimes(categorized_data)

        elif choice == "4":
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


