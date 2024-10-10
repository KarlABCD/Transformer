import pickle
import numpy as np
import csv
import os
from visualization import Visualization

class TrainAnalysis():
    def __init__(self,file):
        self.file = open(file, 'rb')
        return
    def ReadTrainData(self, req_path, req_name):
        traindata = pickle.load(self.file)
        filename = req_path + '/' + req_name + '.csv'
        fieldsname = [i for i in range(len(traindata['Epochs']))]
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            for epoch in traindata['Epochs']:
                writer.writerow([epoch])
                for name_index, name in enumerate(traindata['Name']):
                    if(name == req_name):
                        #Visualization.VisualizePara(traindata['Data'][name_index][epoch], req_name, epoch)
                        writer.writerow({f'parameter shape = {traindata["Data"][epoch][name_index].shape}'})
                        for index, row in enumerate(traindata['Data'][epoch][name_index]):
##                            if(index == 0):
                            writer.writerow(row)
##                            else:
##                                continue
                    else:
                        continue
        return
    def ReadModelInputOutput(self, filename):
        traindata = pickle.load(self.file)
        
        np.set_printoptions(suppress=True, threshold=np.inf)
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            for epoch in traindata['Epochs']:
                writer.writerow([epoch])
                writer.writerow(['input'])
                for index, row in enumerate(traindata['Input'][epoch]):
                    writer.writerow({f'input index = {index}', f'input shape = {row.shape}'})
                    writer.writerow(row)
                writer.writerow(['output'])
                for index, row in enumerate(traindata['Output'][epoch]):
                    writer.writerow({f'output index = {index}', f'output shape = {row.shape}'})
                    writer.writerow(row)
        return

train_analysis = TrainAnalysis('modelrecord/record.pkl')
train_analysis.ReadTrainData('modelanalysis', 'encoder.layers.0.enc_self_attn.W_Q.weight')
#train_analysis.ReadModelInputOutput(os.path.join('modelanalysis', 'model_interface.csv'))