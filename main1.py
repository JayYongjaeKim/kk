import os
import pandas as pd
import random
import re
from datetime import datetime

data_path = '/Users/jake/Downloads/ml-1m' 

user_columns = ['user_id', 'gender', 'age', 'occupation', 'zip']
rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
movie_columns = ['movie_id', 'title', 'genres']

users = pd.read_csv(f'{data_path}/users.dat', sep='::', header=None, names=user_columns, engine='python')
ratings = pd.read_csv(f'{data_path}/ratings.dat', sep='::', header=None, names=rating_columns, engine='python')
movies = pd.read_csv(f'{data_path}/movies.dat', sep='::', header=None, names=movie_columns, engine='python', encoding='latin-1')

movies['movie_year'] = movies['title'].str.extract(r'\((\d{4})\)')
movies['movie_decade'] = movies['movie_year'].astype(float).apply(lambda x: str(int(x - (x % 10))) + 's' if not pd.isnull(x) else None)
movies['title'] = movies['title'].apply(lambda x: re.sub(r'\s*\(\d{4}\)', '', x))
genres_split = movies['genres'].str.split('|')
max_genres = genres_split.apply(len).max()

for i in range(max_genres):
    movies[f'genre{i + 1}'] = genres_split.apply(lambda x: x[i] if i < len(x) else None)
movies.drop('genres', axis=1, inplace=True)
movies.fillna('', inplace=True)
set(movies['genre1'].unique().tolist() + movies['genre2'].unique().tolist() + movies['genre3'].unique().tolist())

ratings['timestamp'] = ratings['timestamp'].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
ratings['rating_year'] = ratings['timestamp'].apply(lambda x: x.split("-")[0])
ratings['rating_month'] = ratings['timestamp'].apply(lambda x: x.split("-")[1])
ratings['rating_decade'] = ratings['rating_year'].astype(int).apply(lambda x: str(x - (x % 10)) + 's')

movies.to_csv(f"{data_path}/movies_prepro.csv", index=False)
ratings.to_csv(f"{data_path}/ratings_prepro.csv", index=False)
users.to_csv(f"{data_path}/users_prepro.csv", index=False)

ratings = ratings[ratings['rating'] >= 3]
ratings['label'] = 1
ratings.drop('rating', axis=1, inplace=True)
user_seen_movies = ratings.groupby('user_id')['movie_id'].apply(list).reset_index()
unique_movies = movies['movie_id'].unique()
unique_users = users['user_id'].unique()
negative_users, negative_movies, negative_labels = [], [], []

for user in unique_users:
    if len(user_seen_movies[user_seen_movies['user_id'] == user]) < 1:
        continue
    user_seen_movie_list = user_seen_movies[user_seen_movies['user_id'] == user]['movie_id'].values[0]
    user_non_seen_movie_list = list(set(unique_movies) - set(user_seen_movie_list))
    sample_pop_size = len(user_seen_movie_list) * 5
    if len(unique_movies) - len(user_seen_movie_list) < len(user_seen_movie_list) * 5:
        sample_pop_size = len(unique_movies) - len(user_seen_movie_list)
    user_negative_movie_list = random.sample(user_non_seen_movie_list, sample_pop_size)

    negative_users += [user for _ in range(len(user_negative_movie_list))]
    negative_movies += user_negative_movie_list
    negative_labels += [0 for _ in range(len(user_negative_movie_list))]

negative_ratings_df = pd.DataFrame({'user_id': negative_users, 'movie_id': negative_movies, 'label': negative_labels})
ratings_df = ratings[['user_id', 'movie_id', 'label']] 
ratings_df = pd.concat([ratings[['user_id', 'movie_id', 'label']], negative_ratings_df], axis=0)
movies_df = movies[['movie_id', 'movie_decade', 'movie_year', 'genre1']]
movies_df.columns = ['movie_id', 'decade', 'movie_year', 'genre']
users_df = users[['user_id', 'gender', 'age', 'occupation', 'zip']]

merge_mlens_data = pd.merge(ratings_df, movies_df, on='movie_id')
merge_mlens_data = pd.merge(merge_mlens_data, users_df, on='user_id')
merge_mlens_data.dropna(inplace=True)
merge_mlens_data = merge_mlens_data[['user_id', 'movie_id', 'movie_decade', 'movie_year', 'genre1', 'gender', 'age', 'occupation', 'zip', 'label']]
merge_mlens_data.to_csv(f'{data_path}/movielens_rcmm_v1.csv', index=False)


users_df = pd.read_csv(f'{data_path}/users_prepro.csv')
ratings_df = pd.read_csv(f'{data_path}/ratings_prepro.csv')
movies_df = pd.read_csv(f'{data_path}/movies_prepro.csv')

ratings_df['label'] = ratings_df['rating'].apply(lambda x : x >=4).astype(int)

ratings_df = ratings_df[['user_id', 'movie_id', 'rating_year','rating_month', 'rating_decade', 'label']]

movies_df = movies_df[['movie_id', 'movie_decade', 'movie_year', 'genre1', 'genre2', 'genre3']]
users_df = users_df[['user_id', 'gender', 'age', 'occupation', 'zip']]

merge_mlens_data = pd.merge(ratings_df, movies_df, on='movie_id')
merge_mlens_data = pd.merge(merge_mlens_data, users_df, on='user_id')
merge_mlens_data.fillna('no', inplace=True)

merge_mlens_data = merge_mlens_data[['user_id', 'movie_id','movie_decade', 'movie_year', 'rating_year', 'rating_month', 'rating_decade', 'genre1','genre2', 'genre3', 'gender', 'age', 'occupation', 'zip', 'label']]


merge_mlens_data.to_csv(f'{data_path}/movielens_rcmm_v2.csv', index=False)

print("전처리 완료")
