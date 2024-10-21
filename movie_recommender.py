# USED WEBSITE BELOW AS GUIDANCE, USING PROJECT TO GET MORE FAMILIAR WITH MACHINE LEARNING
# https://medium.com/@sumanadhikari/building-a-movie-recommendation-engine-using-scikit-learn-8dbb11c5aa4b
##############################################################################################################
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# implement helper function to give us title if we pass in index
##########################################################
def get_title(index):
    
    return movie_list[movie_list.index == index]['title'].values[0]
    
# helper function to give us index if we pass in title
def get_index(title):
    
    return movie_list[movie_list['title'] == title]['index'].values[0]

#########################################################
# uses pandas to read the csv of movie info
movie_list = pd.read_csv('movie_recommender/movie_dataset.csv')
# column names on the movie dataset, looking for these when we want to make connections to curate recommendation
look_for = ['keywords', 'cast', 'genres', 'director']

for i in look_for:
    # fills all NAN values with blank string
    movie_list[i] = movie_list[i].fillna('')
# function to combine values from the look for into a single string
######################################
def combine_looked(row):
    
    try:
        
        return row['keywords'] + ' ' + row['cast'] + ' ' + row['genres'] + ' ' + row['director']
    
    except:
        
        print('Error: ', row )
#########################################

movie_list['combine_looked'] = movie_list.apply(combine_looked, axis = 1)
# creating a count matrix from this new combined column
cv = CountVectorizer() # -> maps a text file to a sparse matrix for mathematical use with cosine analysis
# initalize count matrix with column of combined values
count_matrix = cv.fit_transform(movie_list['combine_looked'])
# compute cosine similarity based on count matrix 
cosine_simholder = cosine_similarity(count_matrix)
# get the movie the recommendation will be based off of
user_choice = input('Enter the title of a movie to see those similar to it: ')
# get the index of the movie with the helper function we defined before
movie_index = get_index(str(user_choice))
# uses cosine to map similarity within the count matrix
similar = list(enumerate(cosine_simholder[movie_index]))
# sorts similar, key = lambda line is an anonymous function that tkes an element x (which is a tuple), 
# nd returns x[1] (second element of the tuple)
# we want the second element of the tuple since we know the most similar movie is the movie itself
# reverse = true sorts the list in descending order
sort_similar = sorted(similar, key = lambda x:x[1], reverse = True)


# print the similar movie names
i = 0
for element in sort_similar:
    
    print(get_title(element[0]))
    i = i+1
    if i>10:
        break