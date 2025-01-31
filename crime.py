from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
import pandas as pd
import plotly.express as px

# Load pretrained BERT model and tokenizer for Named Entity Recognition (NER)
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")

# Load SpaCy for dependency parsing
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

# Map relationships to expanded crime categories
def map_to_crime_categories(relationships):
    # Define expanded crime categories and their associated keywords
    crime_categories = {
        "Murder": ["murder", "kill", "homicide", "manslaughter"],
        "Thievery": ["stole", "theft", "robbed", "burglary", "shoplifting"],
        "Fraud": ["fraud", "scam", "deceit", "embezzlement", "forgery"],
        "Assault": ["assault", "attack", "violence", "battery"],
        "Kidnapping": ["kidnap", "abduct", "hostage"],
        "Arson": ["arson", "fire", "burn"],
        "Drug Trafficking": ["drug", "trafficking", "narcotics"],
        "Cybercrime": ["hacking", "cyber", "phishing", "malware"],
        "Terrorism": ["terrorism", "bombing", "extremism", "radical"],
        "Bribery": ["bribery", "corruption", "kickback"],
        "Sexual Assault": ["rape", "molest", "sexual assault", "harassment"],
        "Extortion": ["extort", "blackmail", "coercion"],
        "Money Laundering": ["launder", "money laundering", "dirty money"],
        "Human Trafficking": ["human trafficking", "smuggling", "exploitation"],
        "Vandalism": ["vandalism", "graffiti", "property damage"],
        "Weapons Offense": ["firearm", "weapon", "gun", "knife"],
        "Smuggling": ["smuggle", "contraband", "illicit trade"],
        "Public Disorder": ["riot", "unrest", "disturbance"],
        "Counterfeiting": ["counterfeit", "fake", "imitation"],
        "Poaching": ["poach", "wildlife crime", "illegal hunting"],
        "Tax Evasion": ["tax evasion", "tax fraud", "evasion"],
        "Perjury": ["perjury", "false testimony", "lying under oath"],
        "Espionage": ["espionage", "spying", "intelligence leak"],
        "Identity Theft": ["identity theft", "impersonation", "identity fraud"],
        "Illegal Immigration": ["illegal immigration", "border crossing", "undocumented"],
        "Domestic Violence": ["domestic violence", "abuse", "intimate partner violence"]
    }

    categorized_data = []
    for rel in relationships:
        # Combine verb and object for easier matching
        action = f"{rel['Relationship']} {rel['Object']}".lower()

        # Check which category the action belongs to
        for category, keywords in crime_categories.items():
            if any(keyword in action for keyword in keywords):
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

# Map relationships to expanded crime categories
categorized_data = map_to_crime_categories(all_relationships)

# Analyze and visualize crime categories
analyze_and_visualize_crimes(categorized_data)

