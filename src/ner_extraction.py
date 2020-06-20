import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import pandas as pd
from tqdm.auto import tqdm
import random


''' Extract named entities using spaCy from all documents '''

def ner_of_article(article, nlp):
    doc = nlp(article)
    return [(X.text, X.label_) for X in doc.ents if relevant_entity(X.label_)]

def relevant_entity(e):
    unwanted = ['ORDINAL', 'CARDINAL', 'QUANTITY', 'MONEY', 'PERCENT', 'TIME', 'DATE']
    return e not in unwanted

def top_n_entities(article, nlp, n):
    doc = nlp(article)
    # doc [(X.text, X.label_) for X in doc.ents]
    # print(article)
    items = [(x.text, x.label_) for x in doc.ents if relevant_entity(x.label_)]
    return Counter(items).most_common(n)

nlp = en_core_web_sm.load()

for i in range(2016, 2021):    
    df_year = pd.read_csv('news-'+str(i)+'.csv')
    # for j in range(10):
    #     x = random.randint(1, df_year.shape[0])
    #     print(top_n_entities(df_year.loc[x]['article'], nlp, 20))

    tqdm.pandas()
    # print(df_year['article'])
    df_year['ner'] = df_year.progress_apply(lambda x: ner_of_article(x['article'], nlp), axis=1)
    df_year.to_csv('news-'+str(i)+'.csv')
    # df['ner'] = df_year.progress_apply(lambda x: top_n_entities(x['article'], nlp, 5), axis=1)

