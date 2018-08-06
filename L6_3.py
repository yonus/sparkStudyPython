import pandas  as pd;
import numpy as np
r_cols = ["user_id","movie_id","rating"]
ratings = pd.read_csv("data/u.data" , sep="\t" , names=r_cols , usecols=range(3), encoding="ISO-8859-1")

m_cols = ["movie_id" ,"title"]
moviews =  pd.read_csv("data/u.item" , sep="|"  ,names=m_cols , usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(moviews,ratings)

print(ratings.head())

movieRatings = ratings.pivot_table(index=["user_id"] , columns=["title"] , values="rating")
print(movieRatings.head())

starWarsRatings = movieRatings["Star Wars (1977)"]

print(starWarsRatings.head())


similarMoviews = movieRatings.corrwith(starWarsRatings)
similarMoviews = similarMoviews.dropna();
df = pd.DataFrame(similarMoviews)
df.head(10)

movieStats =  ratings.groupby("title").agg({"rating":[np.size ,np.mean]})
print(movieStats.head())


populerMovies = movieStats["rating"]["size"] >= 100;
movieStats[]

