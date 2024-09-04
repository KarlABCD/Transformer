import pickle
import numpy as np
from visualization import Visualization
class TrainAnalysis():
    def __init__(self,file):
        self.file = open(file, 'rb')
        return
    def ReadTrainData(self, req_name):
        traindata = pickle.load(self.file)
        for epoch in traindata['Epochs']:
            for name_index, name in enumerate(traindata['Name']):
                if(name == req_name):
                    Visualization.VisualizePara(traindata['Data'][name_index][epoch], req_name, epoch)
                else:
                    continue
        return

train_analysis = TrainAnalysis('output.pkl')
train_analysis.ReadTrainData('encoder.pos_emb.weight')