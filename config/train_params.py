# 语料相关
data_dir = 'corpus'
data_name = 'corpus_data.txt'

# 训练相关
epochs = 100
device_type = 'cpu' # cpu or gpu
dtype = 'bfloat16'
learningrate = 0.0001
# 模型相关的参数
model_config = {'d_k': 64,
                'd_v': 64,
                'd_embedding': 128,
                'n_heads': 2,
                'batch_size' : 5,
                'n_layers': 1,
                'bModelDebug': True}
# 保存格式相关的
bDataRecord = True
ModelOutDir = 'checkpoint'
ModelName = 'checkpoint.pt'
RecordOutDir = 'modelrecord'
RecordName = 'record.pkl'
WorkMode = 'PreTrained'
#WorkMode = 'scratch'
CheckPtNum = 10

# 是否需要打印重要信息
bDebugInfo = False