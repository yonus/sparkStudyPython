import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

r_cols=['user_id', 'movie_id', 'rating'] 
ratings=pd.read_csv("data/u.data", sep="\t", names=r_cols, usecols=range(3), encoding="ISO-8859-1")
m_cols=["movie_id","title"]
movies=pd.read_csv("data/u.item", sep="|", names=m_cols, usecols=range(2),encoding="ISO-8859-1")
ratings=pd.merge(movies, ratings)
movieRatings=ratings.pivot_table(index=["user_id"], columns=['title'], values="rating")

selected_Movie = 'Return of the Jedi (1983)';
movie=movieRatings[selected_Movie]
similarMovies=movieRatings.corrwith(movie)
similarMovies=round(similarMovies.dropna(),2)
movieStats=ratings.groupby('title').agg({'rating': [np.size, np.mean]})

average_similarity_cutoff_dict = {}
for i in range(25,500,25):
    popularMovies=movieStats["rating"]["size"] >i
    df=movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=["similarity"]))
    top_five_similar_movie = df.sort_values(["similarity"], ascending=False)[1:6]
    average_similarity_cutoff_dict[i] = top_five_similar_movie["similarity"].mean()
    print("Most Similar Movies with # of total ratings over ",i,top_five_similar_movie , "\n")

#plot avarage similarity for each cutoff value
lists = sorted(average_similarity_cutoff_dict.items()) # sorted by key, return a list of tuples
x, y = zip(*lists)
plt.plot(x , y,c='r')
plt.plot(x , y,'bo')
plt.xticks(np.arange(min(x),max(x),25))
plt.show()    