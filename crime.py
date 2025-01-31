from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
import pandas as pd
import plotly.express as px

# Load pretrained BERT model and tokenizer for Named Entity Recognition (NER)
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")

# Load SpaCy for dependency parsing and lemmatization
nlp = spacy.load("en_core_web_sm")

# Load dataset from CSV file
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    if 'Text' not in data.columns:
        raise ValueError("The dataset must contain a 'Text' column.")
    return data

# Function to extract relationships
def extract_relationships(text):
    doc = nlp(text)
    relationships = []

    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            if token.pos_ not in ["PRON", "DET", "ADP", "CCONJ"]:
                subject = token.text
                verb = token.head.text
                objects = [
                    child.text
                    for child in token.head.children
                    if child.dep_ in ["dobj", "pobj"] and child.pos_ not in ["ADP", "CCONJ"]
                ]
                for obj in objects:
                    relationships.append({"Subject": subject, "Relationship": verb, "Object": obj})
    return relationships

# Map relationships to crime categories using lemmatization
def map_to_crime_categories(relationships):
    # Define crime categories and their associated keywords
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
        # Combine verb and object for easier matching
        action = f"{rel['Relationship']} {rel['Object']}".lower()

        # Lemmatize the action for comparison
        action_lemmas = [token.lemma_ for token in nlp(action)]

        # Check which category the action belongs to
        for category, keywords in crime_categories.items():
            if any(keyword in action_lemmas for keyword in keywords):
                categorized_data.append({"Category": category, "Action": action})
                break

    return categorized_data

# Analyze and visualize crime categories
def analyze_and_visualize_crimes(categorized_data):
    # Create a DataFrame to count crime categories
    crime_counts = pd.DataFrame(categorized_data).value_counts("Category").reset_index()
    crime_counts.columns = ["Category", "Frequency"]

    # Display the top crime categories in a bar chart
    fig = px.bar(crime_counts, x="Category", y="Frequency", 
                 title="Crime Categories", 
                 labels={"Category": "Crime Category", "Frequency": "Frequency"},
                 text_auto=True)
    fig.update_layout(xaxis_tickangle=45)
    fig.show()

# Example usage
file_path = "Datasets/news_excerpts_parsed.csv"  # Replace with your CSV file path
data = load_dataset(file_path)

# Process and analyze relationships
all_relationships = []
for text in data['Text']:
    relationships = extract_relationships(text)
    all_relationships.extend(relationships)

# Map relationships to crime categories
categorized_data = map_to_crime_categories(all_relationships)

# Analyze and visualize crime categories
analyze_and_visualize_crimes(categorized_data)


