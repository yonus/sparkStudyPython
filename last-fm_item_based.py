import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def createNNCloserItem(similarity_DF , N_closer=10):
  """ 
  assing  N nearest/closer song  for each song
  Parameters
  ---------
  similarity_DF : pandas.DataFrame matrix for similarty scores for paired song
  N_closer : int  determine how much near song are selected 
  """
  if(N_CLOSER < N_closer):
      N_closer = N_CLOSER
  # Create a placeholder items for closer neighbours to an item
  song_nearest_neighbour_DF  = pd.DataFrame(index=similarity_DF.columns,columns=range(1,N_closer+1))
  # Loop through similarity dataframe and fill in neighbouring item names
  for i in range(0,len(similarity_DF.columns)):
    song_nearest_neighbour_DF.iloc[i, :N_closer] = similarity_DF.iloc[0:,i].sort_values(ascending=False)[:N_closer].index    
  return song_nearest_neighbour_DF;

def printTopNSimilarSong(song_nearest_neighbour_DF,song_name , N=10):
   """
   Print N similar Song for given song  from high similarity to low similarity 
   Parameters:
   song_name : str , song name for recommendation
   N : the number of recommemded song  count
   """
   if N_CLOSER < N:
      N = N_CLOSER
   print("Top {0} recommendation for song '{1}'".format(N,song_name))
   print(song_nearest_neighbour_DF.loc[song_name][:N])

#default value of max count of recommadation for each song
N_CLOSER = 20;

lastfmDF  = pd.read_csv("data/LastFM_Matrix.csv")
#drop user column because of item-item based 
songs = lastfmDF.drop("user",axis =1)

#we give tranpose of current songs dataframe as parameter. As result of that , our dataframe is like item-rate matrix 
similarity_matrix = cosine_similarity(songs.T)
similarity_DF = pd.DataFrame(similarity_matrix, columns=(songs.columns), index=(songs.columns))

#build song recommendations for each song
song_nearest_neighbour_DF = createNNCloserItem(similarity_DF,N_CLOSER)

print("Top 10 recommedation for first 5 song ")
print(song_nearest_neighbour_DF.head(5).iloc[:,1:10])


printTopNSimilarSong(song_nearest_neighbour_DF,'travis',5)
