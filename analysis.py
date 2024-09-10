import pickle
import numpy as np
import csv
from visualization import Visualization
class TrainAnalysis():
    def __init__(self,file):
        self.file = open(file, 'rb')
        return
    def ReadTrainData(self, req_name):
        traindata = pickle.load(self.file)
        filename = req_name + '.csv'
        fieldsname = [i for i in range(len(traindata['Epochs']))]
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            for epoch in traindata['Epochs']:
                writer.writerow([epoch])
                for name_index, name in enumerate(traindata['Name']):
                    if(name == req_name):
                        #Visualization.VisualizePara(traindata['Data'][name_index][epoch], req_name, epoch)
                        for index, row in enumerate(traindata['Data'][epoch][name_index]):
                            if(index == 0):
                                writer.writerow(row)
                            else:
                                continue
                    else:
                        continue
        return
    def ReadModelInputOutput(self):
        traindata = pickle.load(self.file)
        filename = 'model_analysis.csv'
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            for epoch in traindata['Epochs']:
                writer.writerow([epoch])
                writer.writerow(['input'])
                writer.writerow(traindata['Input'][epoch])
                writer.writerow(['output'])
                writer.writerow(traindata['Output'][epoch])
        return

train_analysis = TrainAnalysis('output.pkl')
#train_analysis.ReadTrainData('projection.weight')
train_analysis.ReadModelInputOutput()