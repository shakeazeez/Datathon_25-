import pandas as pd


article_data = pd.read_excel("Datasets/news_excerpts_parsed.xlsx")
pdf_data = pd.read_excel("Datasets/wikileaks_parsed.xlsx")


def validate_data(df, expected_columns):
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    if df.isnull().any().any():
        print("Warning: Missing values detected.")
    return df.drop_duplicates().dropna()

article_data = validate_data(article_data, ["Link", "Text"])
pdf_data = validate_data(pdf_data, ["PDF Path", "Text"])
