
# Crime Relationship Extraction and Analysis  

## Overview  
This repository contains a Python-based pipeline for extracting, analyzing, and visualizing crime-related relationships from text data. Using Named Entity Recognition (NER), dependency parsing, and graph-based visualization techniques, this project identifies key entities, their relationships, and maps actions to crime categories.  

## Features  
- **Named Entity Recognition (NER)** using a fine-tuned BERT model (`dbmdz/bert-large-cased-finetuned-conll03-english`).  
- **Dependency parsing** with SpaCy to extract subject-verb-object relationships.  
- **Graph-based visualization** using NetworkX and Matplotlib for relationship mapping.  
- **Crime categorization** based on predefined crime-related keywords.  
- **Bar chart analysis** using Plotly to visualize crime category distribution.  
- **Tokenization and POS tagging** for detailed text analysis.  

## Dependencies  
- `torch`  
- `transformers`  
- `spacy`  
- `networkx`  
- `matplotlib`  
- `pandas`  
- `plotly`  
- `tqdm`  

## Usage  
1. **Prepare your dataset**: Ensure your dataset is in an Excel (`.xlsx`) format with a column named `Text`.  
2. **Run the script**: Execute `EntityFinder.py` or `crime.py` and select from the menu:
   - Extract relationships  
   - Visualize relationships  
   - Analyze crime categories  
   - Tokenize text  
3. **Output files**:  
   - Extracted relationships are saved to `combined_results.xlsx`.  
   - Graphs and charts are displayed for better insights.  

