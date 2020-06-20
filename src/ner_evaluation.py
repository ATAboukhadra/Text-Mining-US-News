import pandas as pd
import en_core_web_sm
import spacy
from spacy import displacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import random
import time
import numpy as np

''' Compare the efficiency of spaCy against NLTK '''

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
def relevant_entity(e):
    unwanted = ['ORDINAL', 'CARDINAL', 'QUANTITY', 'MONEY', 'PERCENT', 'TIME', 'DATE']
    return e not in unwanted

def ner_using_spacy(article):
    doc = nlp(article)
    return doc
    # return [(X.text, X.label_) for X in doc.ents]

def ner_using_nltk(article):
    tokens = word_tokenize(article)
    pos = pos_tag(tokens)
    ner = nltk.ne_chunk(pos)
    return ner

nlp = en_core_web_sm.load()
df = pd.read_csv('data/news-2018.csv')
spacy_time = []
nltk_time = []

for i in range(50):
    x = random.randint(1, df.shape[0]-1)
    article = df.loc[x]['article']
    start = time.time()
    ner_using_spacy(article)
    spacy_elapsed = time.time() - start
    spacy_time.append(spacy_elapsed)
    start = time.time()
    ner_using_nltk(article)
    nltk_elpased = time.time() - start
    nltk_time.append(nltk_elpased)
    print("spacy:", spacy_elapsed, "nltk:", nltk_elpased)
    
print("Spacy average time to recognize entities: ", np.average(np.array(spacy_time)))
print("Nltk average time to recognize entities: ", np.average(np.array(nltk_time)))
