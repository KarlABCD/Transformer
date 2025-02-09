import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import seaborn as sns

def Tensor2Numpy(InputTensor):
    OutputNumpy = InputTensor.detach().numpy()
    return OutputNumpy
def NumPy2Tensor(InputNumpy):
    OutputTensor = torch.from_numpy(InputNumpy)
    return OutputTensor
def PlotOneDimTensor(InputTensor):
    #InputTensorSize = len(InputTensor.size())
    InputTensor = InputTensor.view(-1,InputTensor.shape[-1])
    InputNumpy = Tensor2Numpy(InputTensor)
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
            plt.text(j, i, f'{InputNumpy[i, j]:.2f}', ha='center', va='center', 
                     color='black')
    plt.imshow(InputNumpy, cmap='viridis')
    plt.title(ParaName)
    plt.savefig(f"{index} {ParaName}.jpg",bbox_inches='tight', pad_inches=0)

def ShowHeatmaps(matrices, xlabel, ylabel, figure_id, titles=None, figsize=(2.5, 2.5),
                 cmap='Blues', x_tick_labels=None, y_tick_labels=None):
    """显示矩阵热图"""
    # 设置 matplotlib 支持中文
    plt.rcParams['font.family'] = 'SimHei'  # 在 Windows 上使用黑体
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig = plt.figure(figure_id)
    # 清空图形内容
    fig.clf()
    axes = []
    for i in range(num_rows):
        row_axes = []
        for j in range(num_cols):
            ax = fig.add_subplot(num_rows, num_cols, i * num_cols + j + 1)
            row_axes.append(ax)
        axes.append(row_axes)

    first_pcm = None  # 用于保存第一个子图的 pcm 对象
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            ax.cla()
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if first_pcm is None:
                first_pcm = pcm
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
                if x_tick_labels is not None:
                    # 设置 x 轴刻度位置
                    ax.set_xticks(range(len(x_tick_labels)))
                    # 设置 x 轴刻度标签
                    ax.set_xticklabels(x_tick_labels)
            if j == 0:
                ax.set_ylabel(ylabel)
                if y_tick_labels is not None:
                    # 设置 y 轴刻度位置
                    ax.set_yticks(range(len(y_tick_labels)))
                    # 设置 y 轴刻度标签
                    ax.set_yticklabels(y_tick_labels)
            if titles:
                ax.set_title(titles[j])

    # 添加颜色条
    fig.colorbar(pcm, ax=axes, shrink=0.6)

    # 重绘图形
    plt.draw()
    # 暂停一段时间以便观察更新
    plt.pause(0.1)
