import pandas as pd
import spacy

from rank_bm25 import BM25Okapi
from time import time
from tqdm import tqdm


df = pd.read_csv("data.csv")
nlp = spacy.load("en_core_web_sm")
tok_text = []
text_list = df.text.str.lower().values

for doc in tqdm(nlp.pipe(text_list, disable=["tagger", "parser", "ner", "lemmatizer"])):
    tok = [t.text for t in doc if t.is_alpha]
    tok_text.append(tok)

bm25 = BM25Okapi(tok_text)

query = "Flood Defence"
tokenized_query = query.lower().split()

start = time()
results = bm25.get_top_n(tokenized_query, df.text.values, n=3)
end = time()

print(f"Searched 50,000 records in {round(end - start, 3)} seconds\n.")

for res in results:
    print(res)

