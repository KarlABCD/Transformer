import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import axes3d
import numpy as np

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