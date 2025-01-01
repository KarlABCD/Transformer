# 训练相关
epochs = 60
device_type = 'cpu' # cpu or gpu
dtype = 'bfloat16'
learningrate = 0.0001
# 模型相关的参数
model_config = {'d_k': 64,
                'd_v': 64,
                'd_embedding': 128,
                'n_heads': 1,
                'batch_size' : 5,
                'n_layers': 1}
# 保存格式相关的
bDataRecord = True
ModelOutDir = 'checkpoint'
ModelName = 'checkpoint.pt'
RecordOutDir = 'modelrecord'
RecordName = 'record.pkl'
WorkMode = 'PreTrained'
#WorkMode = 'scratch'
CheckPtNum = 10
