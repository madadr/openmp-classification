from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import datetime
import csv

start = datetime.datetime.now()
dataset = []
genres = []
with open('csv/letter-recognition.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
     for row in spamreader:
         dataset.append(row[1:])
         genres.append(row[:1][0])

print('Dataset: ', dataset)
print('Genres: ', genres)
trainingSetSize = int(len(dataset)*0.9)
print('\n', trainingSetSize)

# MinMaxScaler

minMaxScaler = MinMaxScaler()
minMaxDataset = minMaxScaler.fit_transform(dataset)
print(minMaxDataset)

#StandardScaler

standardScaler = StandardScaler()
standarizedDataset = standardScaler.fit_transform(dataset)
print(standarizedDataset)


#KNN classification algorithm

# Without standarization
knn = KNeighborsClassifier(n_neighbors=1)
trainingDataset = dataset[:trainingSetSize]
knn.fit(trainingDataset, genres[:trainingSetSize])
prediction = knn.predict(dataset[trainingSetSize:])
print(prediction)

accuracy = accuracy_score(prediction, genres[trainingSetSize:])
print('Accuracy: ', accuracy)

#MinMaxScaler
trainingDataset = minMaxDataset[:trainingSetSize]
knn.fit(trainingDataset, genres[:trainingSetSize])
minMaxPrediction = knn.predict(minMaxDataset[trainingSetSize:])
print(minMaxPrediction)

accuracy = accuracy_score(minMaxPrediction, genres[trainingSetSize:])
print('MinMax accuracy: ', accuracy)

#StandardScaler
trainingDataset = standarizedDataset[:trainingSetSize]
knn.fit(trainingDataset, genres[:trainingSetSize])
standarizedPrediction = knn.predict(standarizedDataset[trainingSetSize:])
print(standarizedPrediction)

accuracy = accuracy_score(standarizedPrediction, genres[trainingSetSize:])
print('Standarization accuracy: ', accuracy)

end = datetime.datetime.now()
elapsed = end - start
print('Time: ', elapsed)