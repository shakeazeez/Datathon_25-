from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go  # For interactive visualization
import plotly.io as pio
pio.renderers.default = 'browser'


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

# Enhanced visualization with Plotly
def visualize_relationships_interactive(relationships):
    if not relationships:
        print("No relationships to visualize.")
        return

    # Create directed graph
    G = nx.DiGraph()

    # Add edges for relationships with labels
    for rel in relationships:
        G.add_edge(rel["Subject"], rel["Object"], label=rel["Relationship"])

    # Extract nodes and edges
    edge_x = []
    edge_y = []
    edge_labels = []
    pos = nx.spring_layout(G)
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_labels.append(edge[2]['label'])

    # Add edges to Plotly figure
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Add nodes to Plotly figure
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=20,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    # Build the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Entity Relationship Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=30),
                        annotations=[
                            dict(
                                text="Hover over nodes and edges to explore relationships",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002
                            )
                        ],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )

    fig.show()

# Example usage
file_path = "Datasets/news_excerpts_parsed.csv"  # Replace with your CSV file path
data = load_dataset(file_path)

# Process and visualize relationships
all_relationships = []
for text in data['Text']:
    relationships = extract_relationships(text)
    all_relationships.extend(relationships)

visualize_relationships_interactive(all_relationships)
