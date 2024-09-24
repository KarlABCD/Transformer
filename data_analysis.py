import numpy as np
import pickle
import matplotlib.pyplot as plt

file = open('output.pkl', 'rb')
traindata = pickle.load(file)
np.set_printoptions(suppress=True, threshold=np.inf)
fig, ax = plt.subplots()
for epoch in traindata['Epochs']:
    for index, row in enumerate(traindata['Input'][epoch]):
        for i in range(len(row)):
            for j in range(len(row[i])):
              if(row[i][j] == 1 or row[i][j] == 2):
                  #print(traindata['Output'][epoch][0][i][j])
                  ax.plot( traindata['Output'][epoch][0][i][j])
                  print(traindata['Output'][epoch][1])
plt.show()