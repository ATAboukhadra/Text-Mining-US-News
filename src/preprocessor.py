import pandas as pd
import sys
import numpy as np
from tqdm.auto import tqdm

# np.set_printoptions(threshold=np.inf)

years = [2016,2017,2018,2019,2020]

""" Split based on year """

# df = pd.read_csv('all-the-news.csv', escapechar='\\', error_bad_lines=False)
# for i in range(5):
#     df_year = df[df['year'] == years[i]]
#     df_year.to_csv('news-'+str(years[i])+'.csv')

""" Remove Unnecessary Columns """

# for i in range(5):
#     df_year = pd.read_csv('news-'+str(years[i])+'.csv')
#     df_year = df_year.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'url'])
#     df_year.to_csv('news-'+str(years[i])+'.csv')


""" Remove articles with nan values"""

# for i in range(5):
#     df_year = pd.read_csv('news-'+str(years[i])+'.csv')
#     df_year = df_year[df_year['article'].notna()]
#     df_year.to_csv('news-'+str(years[i])+'.csv')
#     print(df_year.isna().sum())

""" Statistics """

# for i in range(5):
#     df_year = pd.read_csv('news-'+str(years[i])+'.csv')
#     cols = df_year.columns
#     for col in cols:
#         print(col, df_year[col].dtype)
#         if col == 'publication' or col == 'section' or col == 'month':
#             print(col, df_year[col].unique())

""" Processing the sentiment column """
tqdm.pandas()
for i in range(5):
    df_year = pd.read_csv('news-'+str(years[i])+'.csv')
    df_year['sentiment-svm'] = df_year['sentiment-svm'].progress_apply(lambda x: int(eval(x)[0]))
    df_year.to_csv('news-'+str(years[i])+'.csv', index=False)
#     for col in cols:
#         print(col, df_year[col].dtype)
#         if col == 'publication' or col == 'section' or col == 'month':
#             print(col, df_year[col].unique())


""" Number in each section """
# df = pd.read_csv('news-2016.csv')
# print(df.columns)
# print(df['section'].unique())
# cates = df.groupby('section')
# grouped = df.groupby('section', sort=False).size()
# grouped.sort_index(ascending=False)
# print(grouped)
# print("total categories:", cates.ngroups)
# print(cates.size())

# df = df.groupby('section').size().nlargest(50) #.reset_index(name='count').sort_values(['count'], ascending=False).head(10)
# print(df)

# def group_and_save():