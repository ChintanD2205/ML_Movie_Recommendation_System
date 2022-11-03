# Movie Recommendation
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
# TfidfVectorizer - This is used to convert text data into numerical values
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("C:\\Users\\chint\\Downloads\\movies.csv") # load dataset
"""print(movies.head())# print top 5 data from movies dataframe
print(movies.info())  # print the info of the dataframe
print(movies.shape)  # print the columns, its datatype, if its null or not , how many data are present in the particular column
"""

features = ['genres', 'keywords', 'tagline', 'title', 'cast', 'director']  # normal list creation
"""print(features)
print(movies.isna().sum())  # checks for null values in movies dataframe
print(movies[features].head())  # prints the top 5 values in movies dataframe pertaining to mentioned features
print(movies[features].isna().sum())  # prints the total no. of null values (if any) in the movies dataframe pertaining to mentioned features
"""
for feature in features:  # normal for loop
    movies[feature] = movies[feature].fillna('')  # fills all the null spaces with a white character space for mentioned features
# print(movies.head())
# print(movies[features].isna().sum())  # to check whether all previous mentioned null spaces have been filled or not; if filled 0 would appear

combined_features = movies['genres']+' '+movies['keywords']+' '+movies['tagline']+' '+movies['title']+' '+movies['cast']+' '+movies['director'] # combine all mentione features for easy access
# print(combined_features)

# To use the similarity we must find cosine similarity of preferred data
# for this we must convert the text based data to numeric data for this purpose the TfidfVectorizer library is used
vectorize = TfidfVectorizer() # initialize a variable with the TfidfVectorizer to call its functions further for model testing and training
feature_vector = vectorize.fit_transform(combined_features) # this fits the combined_features into the feature_vector model and then transforms it to the corresponding numeric value
print(feature_vector.shape) # print the no. of columns and rows
print(feature_vector) # print the numeric value

cos_sim = cosine_similarity(feature_vector) # finds the cosine similarity of feature_vectors as per inherent formula
print(cos_sim) # print cosine similarity matrix

fav_movie = input("Enter the name of your favourite movie : ")

# Creating a list with all the movie names given in the dataset
list_of_all_titles = movies['title'].tolist()
print(list_of_all_titles)

# to find a close match we'll use the diff lib
find_match = difflib.get_close_matches(fav_movie, list_of_all_titles)
print(find_match)
print(len(find_match))

best_match = find_match[0] # we have taken 0th index because the best match would be the first movie
print(best_match)

# Finding the index of the movie with the title
index_of_the_movie = movies[movies.title == best_match]['index'].values[0]
print(index_of_the_movie) # prints index at which the best match is encountered

similarity_score = list(enumerate(cos_sim[index_of_the_movie]))
print(similarity_score)

sort = sorted(similarity_score, key= lambda x:x[1], reverse=True)
print(sort)

print('Movies suggested for you : \n')
i = 1
for movie in sort:
  index = movie[0]
  title_from_index = movies[movies.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1



