#importing libararies
import numpy as np
import pandas as pd
import ast

#importig dataframe
movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

movies.head(5)
credits.head(5)

data=movies.merge(credits, how='left', on='title')
#normalize function gives relative frequency
#data['original_language'].value_counts(normalize=True)*100  
#it shows that english is dominant
#data['original_language'].nunique()

#Taking relevant columns only
movies=data[['movie_id','title','overview','genres','keywords','cast','crew']]

#data preprocessing
movies.isnull().sum()
movies.dropna(inplace=True)
movies.isnull().sum()
movies.duplicated().sum()

#genres, keywords,cast, crew are in json format
#extracting data from json for keywords and and genres columns

def extractor(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

movies['genres']=movies['genres'].apply(extractor)
movies['keywords']=movies['keywords'].apply(extractor)
#movies.genres
#movies.keywords
movies.iloc[0].cast #as it is very long list of dictionaries
#we will take only top3 actors in our list who are related to movies

#creating extractor for cast column
def extractor2(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
          break
    return L
movies['cast']=movies['cast'].apply(extractor2)

movies.head(5)

#extract director name
def director_fetch(obj):
  L=[]
  for i in ast.literal_eval(obj):
    if i['job']=='Director':
      L.append(i['name'])
      break
  return L

movies['crew']=movies['crew'].apply(director_fetch)

movies.head(5)
#converting overwiew in list so that we have all the words
movies.overview=movies.overview.apply(lambda x:x.split())

#remove space from the names
#use transformation
movies.genres=movies.genres.apply(lambda x: [i.replace(" ","") for i in x])
movies.keywords=movies.keywords.apply(lambda x: [i.replace(" ","") for i in x])
movies.cast=movies.cast.apply(lambda x: [i.replace(" ","") for i in x])
movies.crew=movies.crew.apply(lambda x: [i.replace(" ","") for i in x])
movies.head(5)
#creating tags for each movie to identify each movie
movies['tags']= movies['overview'] + movies['genres']+movies['keywords'] + movies['cast'] + movies['crew']
#creating new dataframe
new_df=movies[['movie_id','title','tags']]
new_df.head(5)

#now convert list to string for tags
new_df['tags']= new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags']=new_df['tags'].apply(lambda x: x.lower())
new_df.head()

#remove dulicate same root words
#use stemming to convert love, loving, loved to love only
import nltk
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
    
  return " ".join(y)

new_df['tags']=new_df['tags'].apply(stem)

#convert text into vectors to see the similarity between two movies
#bag of words
#in vectorization dont use stop words
#stop words are those words which are used for sentence formation
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000, stop_words='english')

vectors=cv.fit_transform(new_df['tags']).toarray()
cv.get_feature_names_out()

from sklearn.metrics.pairwise import cosine_similarity

similarity=cosine_similarity(vectors)
#similarity[1]

#sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]
#creating a function to recommend movies 5 movies basis of input
def movie_recommend(movie):
  movie_index= new_df[new_df['title']== movie].index[0]
  distances=similarity[movie_index]
  movie_list= sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

  for i in movie_list:
    print(new_df.iloc[i[0]].title)

movie=input("Pls input the movies")
movie_recommend(movie)
