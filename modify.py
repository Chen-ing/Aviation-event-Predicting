#coding:utf-8
import numpy
import numpy
import string
from collections import Counter
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense

numpy.random.seed(1)

dataset = numpy.genfromtxt('data/data.csv', delimiter = ',', dtype = None)
x = []
y = []
xwords = []
rawdata = []
a = []
temp = []

#parse data
for i in dataset:
 #   print type(i)
#    print i

    '''i
    j = str(i)
    k = j.split("'")
    '''
    a.append(i[0])
    a.append(i[1])
    a.append(i[2])
    '''
    rawdata.append(i)
    '''
    #print a
    rawdata.append(a)
    a = []
rawdata.pop(0)
#print rawdata

for i in rawdata:
    for j in rawdata:
        if i[0] == j[0] and i != j:
            i[1] += " " + j[1]
            del rawdata[rawdata.index(j)]

numpy.random.shuffle(rawdata)
#print rawdata
str1 = ''
for i in rawdata:
    xwords.append(i[1].translate(None, string.punctuation))
    y.append(i[2])
    str1 += ' ' + i[1].translate(None, string.punctuation)
#print str1
counts = Counter(str1.split())
numwords = len(counts)
dict = {}
counter = 100
for i in counts:
    dict[i] = counter
    counter += 1
#print dict

for i in xwords:
    a = i.split()
    #print a
    b = []
    for j in a:
 #       print j
        b.append(dict[j])
    x.append(b)

#print x
#print y

longest_sample_len = len(max(x, key = len))
half = len(x) / 2

x = sequence.pad_sequences(x, maxlen = longest_sample_len)
#numpy.array(x)
#numpy.array(y)
train_x = x[:half]
train_y = y[:half]
test_x = x[half:]
test_y = y[half:]
#print y[:20]

for epochs in [10, 32, 100, 500, 1000]:
    for k in range(5, 50, 5):
        model = Sequential()
        model.add(Embedding(3000, 32))
        model.add(LSTM(32))
        model.add(Dense(32, activation = 'tanh'))
        model.add(Dense(1, activation = 'sigmoid'))

        model.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = 
                     ['accuracy'])
        model.fit(numpy.array(train_x), numpy.array(train_y), epochs = epochs, batch_size = k, verbose = 0)
        scores = model.evaluate(numpy.array(test_x), numpy.array(test_y), batch_size = 32, verbose = 0)
        print("Epochs: " + str(epochs) + " | Batch size: "+ str(k)+ " |  Accuracy:" +
              str(round(scores[1] * 100, 2)) + "%\n")

