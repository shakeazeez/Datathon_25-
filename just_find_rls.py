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
nlp = spacy.load("en_core_web_sm")  # Using transformer-based model for better accuracy

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

        # Ensure the entity list is not empty before modifying
        if word.startswith("##") and current_entity:
            current_entity[-1] += word[2:]  # Append subword without ##
        else:
            if current_entity:  # Store previous entity before starting a new one
                entities[current_label].add(clean_entity(" ".join(current_entity)))
            current_entity = [word]
            current_label = label

    if current_entity:
        entities[current_label].add(clean_entity(" ".join(current_entity)))  # Store last entity

    # Manually add economic terms, numbers, and missed proper nouns
    doc = nlp(text)
    for ent in doc.ents:  # Use spaCy's built-in entity recognition for better accuracy
        if ent.label_ in {"PERSON", "ORG", "GPE", "MONEY", "PERCENT", "DATE", "NUM"}:
            entities[ent.label_].add(clean_entity(ent.text))

    # Convert sets to lists to return unique values
    #print(entities)
    return entities

# def extract_relationships(text, entities):
#     """Extract meaningful relationships based on verb phrases and dependencies."""
#     doc = nlp(text)
#     relationships = set()

#     for token in doc:
#         if token.pos_ == "VERB":  # Use verbs to build relationships
#             subjects = [child.text for child in token.children if child.dep_ in {"nsubj", "nsubjpass", "agent"} and child.text in entities]
#             objects = [child.text for child in token.children if child.dep_ in {"dobj", "pobj", "attr", "prep", "xcomp", "ccomp"} and child.text in entities]
            
#             # Capture prepositional phrases related to the verb
#             prep_phrases = []
#             for child in token.children:
#                 if child.dep_ == "prep":
#                     prep_obj = next((grandchild.text for grandchild in child.children if grandchild.text in entities), None)
#                     if prep_obj:
#                         prep_phrases.append(f"{child.text} {prep_obj}")
            
#             for subject in subjects:
#                 for obj in objects:
#                     phrase = f"{token.lemma_} {obj}"
#                     if prep_phrases:
#                         phrase += " " + " ".join(prep_phrases)
#                     relationships.add((subject, phrase))
    
#     # Capture implicit relationships using sentence structure
#     # entity_list = list(entities.keys())
#     # for i in range(len(entity_list)):
#     #     for j in range(i + 1, len(entity_list)):
#     #         if entity_list[i] != entity_list[j]:
#     #             relationships.add((entity_list[i], f"associated with {entity_list[j]}"))
    
#     return [f"{subj} {rel}" for subj, rel in relationships]
import spacy
from collections import defaultdict

def extract_relationships(text, entities):
    """Extract nested relationships with improved entity mapping and clearer structure."""
   
    doc = nlp(text)
    relationships = set()
    
    # Flatten entity sets into a dictionary for better lookup
    entity_dict = {}
    for label, entity_set in entities.items():
        for entity in entity_set:
            for word in entity.split():
                entity_dict[word.lower()] = entity  # Map each word to its full entity name
    
    def get_full_entity(word):
        """Retrieve the full entity name if available."""
        return entity_dict.get(word.lower(), word)
    
    def extract_nested_relationships(token):
        """Recursively extract valid nested relationships from dependent clauses."""
        subjects = [get_full_entity(child.text) for child in token.children 
                    if child.dep_ in {"nsubj", "nsubjpass", "agent"}]
        objects = [get_full_entity(child.text) for child in token.children 
                   if child.dep_ in {"dobj", "pobj", "attr", "prep", "xcomp", "ccomp"}]
        
        nested_rels = []
        for subject in subjects:
            for obj in objects:
                if subject and obj:
                    relationship = f"{subject} {token.lemma_} {obj}"
                    nested_rels.append(relationship)
                    
                    # Recursively extract relationships from dependent clauses
                    for child in token.children:
                        if child.pos_ in {"VERB", "NOUN", "ADJ"}:
                            nested_rels.extend(extract_nested_relationships(child))
        
        return nested_rels
    
    for token in doc:
        if token.pos_ in {"VERB", "NOUN", "ADJ"} and token.lemma_ not in {"be", "have", "do", "say", "make", "go"}:  # Exclude auxiliary verbs
            relationships.update(extract_nested_relationships(token))
    
    return list(relationships)

# def extract_relationships(text, entities):
#     """Extract nested relationships using dependency parsing and verb phrases."""
#     doc = nlp(text)
#     relationships = set()
    
#     # Build a mapping for full names of PERSON entities
#     full_names = {name for name in entities.get("PERSON", [])}
    
#     def get_full_name(word):
#         return max((name for name in full_names if word in name.split()), key=len, default=word)
    
#     for token in doc:
#         if token.pos_ in {"VERB", "NOUN", "ADJ"}:  # Meaningful parts of speech
#             subjects = [get_full_name(child.text) for child in token.children 
#                         if child.dep_ in {"nsubj", "nsubjpass", "agent"}]
#             objects = [get_full_name(child.text) for child in token.children 
#                        if child.dep_ in {"dobj", "pobj", "attr", "prep", "xcomp", "ccomp"}]
            
#             # Capture nested relationships
#             prep_phrases = []
#             for child in token.children:
#                 if child.dep_ == "prep":
#                     prep_obj = next((get_full_name(grandchild.text) for grandchild in child.children if grandchild.text in entities), None)
#                     if prep_obj:
#                         prep_phrases.append(f"{child.text} {prep_obj}")
            
#             for subject in subjects:
#                 for obj in objects:
#                     phrase = f"{token.lemma_} {obj}"
#                     if prep_phrases:
#                         phrase += " " + " ".join(prep_phrases)
#                     relationships.add((subject, phrase))
    
#     return [f"{subj} {rel}" for subj, rel in relationships]

def process_dataframe(df, source):
    """Process each row of a dataframe to extract entities and relationships."""
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row['Text'] if 'Text' in row else row.iloc[0]  # Ensure correct column reference
        entities = extract_entities(text)
        relationships = extract_relationships(text, entities)
        results.append({"source": source, "text": text, "entities": list(entities.values()), "relationships": relationships})
    
    return results

# Load datasets
df1 = pd.read_excel("Datasets/news_excerpts_parsed.xlsx", sheet_name="Sheet1")
df2 = pd.read_excel("Datasets/wikileaks_parsed.xlsx", sheet_name="Sheet1")

df1_results = process_dataframe(df1, "news_excerpts")
df2_results = process_dataframe(df2, "wikileaks")

# Combine results into a single DataFrame
final_df = pd.DataFrame(df1_results + df2_results)
final_df.to_excel("combined_results_fixed_v2.xlsx", index=False)

print("Processing complete. Results saved to combined_results_fixed_v2.xlsx")
