from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("MaxTemperatures")
sc = SparkContext.getOrCreate();

def parseLine(line):
    fields = line.split(',')
    stationID = fields[0]
    entryType = fields[2]
    temperature = float(fields[3]) * 0.1 * (9.0 / 5.0) + 32.0
    return (stationID, entryType, temperature)

lines = sc.textFile("file:/home/pasa/sparkdevelopment/data/1800.csv")
parsedLines = lines.map(parseLine)
maxTemps = parsedLines.filter(lambda x: "TMAX" in x[1])

stationTemps = maxTemps.map(lambda x: (x[0], x[2]))

#reduce by max TMAX value of each station;
maxTemps = stationTemps.reduceByKey(lambda x, y: max(x,y))

# find station that have highest TMAX value all time
highestTemperatureStation = maxTemps.max(key = lambda x: x[1]);


#get all temperature values of highestTemperatureStation
temperatureValues = stationTemps.filter(lambda x : x[0] == highestTemperatureStation[0]).map(lambda x : x[1]);

#show first 20 record
print(temperatureValues.take(20));

#print avarage of temperature values
print(temperatureValues.mean())
