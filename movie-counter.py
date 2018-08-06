from pyspark import SparkConf, SparkContext
import codecs

def loadMovieNames():
    movieNames = {}
    with open("data/u.item") as f:
    # The following line may need to handle
    # encoding issues on some Ubuntu systems:
    #with codecs.open("ml-100k/u.ITEM", "r", encoding='utf-8', errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames


conf = SparkConf().setMaster("local").setAppName("PopularMovies")
sc = SparkContext.getOrCreate();

nameDict = sc.broadcast(loadMovieNames())

lines = sc.textFile("file:/home/pasa/sparkdevelopment/data/u.data")
movies = lines.map(lambda x: (int(x.split()[1]), 1))
movieCounts = movies.reduceByKey(lambda x, y: x + y)

flipped = movieCounts.map(lambda x, y : (y,x));
sortedMovies = flipped.sortByKey()

sortedMoviesWithNames = sortedMovies.map(lambda count, movie : (nameDict.value[movie], count))

results = sortedMoviesWithNames.collect()

for result in results:
    print (result);
