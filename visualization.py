import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import seaborn as sns

class Visualization:    
    def Tensor2Numpy(InputTensor):
        OutputNumpy = InputTensor.detach().numpy()
        return OutputNumpy
    def NumPy2Tensor(InputNumpy):
        OutputTensor = torch.from_numpy(InputNumpy)
        return OutputTensor
    def PlotOneDimTensor(InputTensor):
        #InputTensorSize = len(InputTensor.size())
        InputTensor = InputTensor.view(-1,InputTensor.shape[-1])
        InputNumpy = Visualization.Tensor2Numpy(InputTensor)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(np.arange( InputNumpy.shape[1]), np.arange(InputNumpy.shape[0]))
        x = x.ravel()
        y = y.ravel()
        InputNumpy = InputNumpy.ravel()
        bottom=np.zeros_like(x)
        width=height=1#每一个柱子的长和宽
        ax.bar3d(x, y, bottom, width, height, InputNumpy, shade=True)#
        ax.set_xlabel('feature', fontdict={'fontname': 'FangSong', 'fontsize': 16})
        ax.set_ylabel('num', fontdict={'fontname': 'FangSong', 'fontsize': 16})
        ax.set_zlabel('values', fontdict={'fontname': 'FangSong', 'fontsize': 16})
        ax.set_aspect('equalxy', 'box')
        plt.savefig("sine_wave.png")
        #plt.show()
    def VisualizeAttention(source_sentences, predicted_sentences, attn_weights):
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(attn_weights, annot = True, cbar = False,
                         xticklabels=source_sentences.split(),
                         yticklabels=predicted_sentences,cmap="Greens")
        plt.xlabel("源序列")
        plt.ylabel("目标序列")
        plt.show()
    def VisualizePara(InputNumpy, ParaName, index):
        fig = plt.figure(figsize=(InputNumpy.shape[1], InputNumpy.shape[0]))
        for i in range(InputNumpy.shape[0]):
            for j in range(InputNumpy.shape[1]):
                plt.text(j, i, f'{InputNumpy[i, j]:.2f}', ha='center', va='center', color='black')
        plt.imshow(InputNumpy, cmap='viridis')
        plt.title(ParaName)
        plt.savefig(f"{index} {ParaName}.jpg",bbox_inches='tight', pad_inches=0)