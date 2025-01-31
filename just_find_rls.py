import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
import spacy
from tqdm import tqdm
import re
from collections import defaultdict

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pretrained BERT model and tokenizer for Named Entity Recognition (NER)
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple", device=0 if device.type == "cuda" else -1)

# Load SpaCy model for dependency parsing
nlp = spacy.load("en_core_web_trf")  # Transformer-based model for better accuracy

def clean_entity(entity):
    """Ensure multi-word entities stay together, especially names and numerical values."""
    entity = entity.strip()
    entity = re.sub(r'\s+(billion|million|thousand|percent|%)', r' \1', entity)  # Keep numerical phrases together
    return entity

def extract_entities(text):
    """Extract named entities from text, ensuring names stay together and improving entity consistency."""
    ner_results = ner_pipeline(text)
    entities = defaultdict(set)
    current_entity = []
    current_label = None
    
    for entity in ner_results:
        word = entity["word"]
        label = entity["entity_group"]
        
        if word.startswith("##") and current_entity:
            current_entity[-1] += word[2:]  # Append subword without ##
        else:
            if current_entity:
                entities[current_label].add(clean_entity(" ".join(current_entity)))
            current_entity = [word]
            current_label = label
    
    if current_entity:
        entities[current_label].add(clean_entity(" ".join(current_entity)))  # Store last entity
    
    doc = nlp(text)
    for ent in doc.ents:
        entities[ent.label_].add(clean_entity(ent.text))
    
    return entities

def extract_relationships(text, entities):
    """Extract relationships between entities based on verb phrases, dependencies, and contextual analysis."""
    doc = nlp(text)
    relationships = set()
    entity_dict = {word.lower(): entity for label, entity_set in entities.items() for entity in entity_set for word in entity.split()}
    
    def get_full_entity(word):
        return entity_dict.get(word.lower(), word)
    
    for token in doc:
        if token.pos_ in {"VERB", "NOUN", "ADJ"}:
            subjects = [get_full_entity(child.text) for child in token.children if child.dep_ in {"nsubj", "nsubjpass", "agent"}]
            objects = [get_full_entity(child.text) for child in token.children if child.dep_ in {"dobj", "pobj", "attr", "prep", "xcomp", "ccomp"}]
            
            for subject in subjects:
                for obj in objects:
                    if subject and obj:
                        relationship = f"{subject} {token.lemma_} {obj}"
                        relationships.add(relationship)
    
    return list(relationships)

def process_dataframe(df, source):
    """Process each row of a dataframe to extract entities and relationships."""
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row['Text'] if 'Text' in row else row.iloc[0]
        entities = extract_entities(text)
        relationships = extract_relationships(text, entities)
        results.append({"source": source, "text": text, "entities": list(entities.values()), "relationships": relationships})
    
    return results

# Load datasets
#df1 = pd.read_excel("Datasets/news_excerpts_parsed.xlsx", sheet_name="Sheet1")
df2 = pd.read_excel("Datasets/wikileaks_parsed.xlsx", sheet_name="Sheet1")

#df1_results = process_dataframe(df1, "news_excerpts")
df2_results = process_dataframe(df2, "wikileaks")

# Combine results into a single DataFrame
#final_df = pd.DataFrame(df1_results + df2_results)
final_df = pd.DataFrame(df2_results)
final_df.to_excel("combined_results_fixed_v3.xlsx", index=False)

print("Processing complete. Results saved to combined_results_fixed_v3.xlsx")
