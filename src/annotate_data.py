import pandas as pd
from sklearn.metrics import classification_report
import numpy as np

# ----------- Step 1: Sample -------------
# df = pd.read_csv("data/news-2019.csv")
# df_sampled = df.sample(n = 90) 
# df_sampled[['Unnamed: 0', 'article', 'sentiment-svm', 'sentiment']].to_csv('data/sampled-articles-2019.csv', index=False)
# print(df_sampled.head())
# ----------- Step 2: Label-----------

# ----------- Step 3: Process--------
def prepare_sentiments(year, thr=0.5):
    labeled_df = pd.read_csv(f"data/sampled-{year}-annotated.csv")
    labeled_df = labeled_df.rename(columns={"Unnamed: 1": "article", "Unnamed: 2":"sentiment-svm", "Unnamed: 3":"sentiment"})
    labeled_df = labeled_df.iloc[1:-1]
    labeled_df = labeled_df[labeled_df['Classification']!='Neutral']
    labeled_df = labeled_df[labeled_df['Confidence'] > thr]
    df_cats = ['Negative', 'Positive']
    y_true = labeled_df.Classification.astype("category").cat.codes.values
    y_svm = labeled_df['sentiment-svm'].values.astype('int')
    y_lr = labeled_df['sentiment'].values.astype('int')
    return y_true, y_svm, y_lr


y_true_2018, y_svm_2018, y_lr_2018 = prepare_sentiments(2018, 0.8)
y_true_2019, y_svm_2019, y_lr_2019 = prepare_sentiments(2019, 0.8)
y_true_2020, y_svm_2020, y_lr_2020 = prepare_sentiments(2020, 0.8)
y_true = np.concatenate((y_true_2018, y_true_2020, y_true_2019))
y_svm = np.concatenate((y_svm_2018, y_svm_2020, y_svm_2019))
y_lr = np.concatenate((y_lr_2018, y_lr_2020, y_lr_2019))

# ----------- Step 4: Evaluate---------
print("Classification Report for SVM Analyzer\n", classification_report(y_true, y_svm))
print("Classification Report for Linear Regression Analyzer\n", classification_report(y_true, y_lr))