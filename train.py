import torch 
import torch.nn as nn
import torch.optim as optim
from model import Transformer, greedy_decoder, forward_hook
from prepare import TranslationCorpus
from visualization import Visualization
import pickle
import copy
import os

# GPU
device_type = 'gpu'
dtype = 'bfloat16'
ModelOutDir = 'checkpoint'
ModelName = 'checkpoint.pt'
RecordOutDir = 'modelrecord'
RecordName = 'record.pkl'
WorkMode = 'PreTrained'
#WorkMode = 'scratch'
sentences = [
          ['咖哥 喜欢 小冰', 'KaGe likes XiaoBing'],
            ['我 爱 学习 人工智能', 'I love studying AI'],
            ['深度学习 改变 世界', 'DL changed the world'],
            ['自然 语言 处理 很 强大', 'NLP is so powerful'],
            ['神经网络 非常 复杂', 'Neural-Nets are complex']]

corpus = TranslationCorpus(sentences)
criterion = nn.CrossEntropyLoss()
if (os.path.exists(os.path.join(ModelOutDir, ModelName)) and WorkMode == 'PreTrained'):
    PreCheckPt = torch.load(os.path.join(ModelOutDir, ModelName))
    model = PreCheckPt['model']
    optimizer = PreCheckPt['optimizer']
else:
    model = Transformer(corpus)
    optimizer = optim.Adam(model.parameters(),lr=0.0001)

epochs = 100
bDataRecord = True
if device_type == 'cuda' and torch.cuda.is_available():
    model.to('cuda')

# 数据记录配置
if bDataRecord:
    os.makedirs(RecordOutDir, exist_ok=True)
    file = open(os.path.join(RecordOutDir, RecordName), 'wb')
    Epochs = []
    Names = []
    DataList = [[] for _ in range(len(list(model.named_parameters())))]
    Data = []
    ModelInput =  [[] for _ in range(0,epochs)]
    ModelOutput =  [[] for _ in range(0,epochs)]
    forward_hook.inputs = []
    forward_hook.outputs = []

for epoch in range(epochs):
    optimizer.zero_grad()
    #model_parameter = list(model.parameters())
    enc_inputs, dec_inputs, target_batch = corpus.make_batch(3, device_type)
    #print([corpus.src_idx2word[value] for i in range(enc_inputs.size(0)) for value in enc_inputs[i].tolist()])
    #print([corpus.tgt_idx2word[value] for i in range(dec_inputs.size(0)) for value in dec_inputs[i].tolist()])
    #print([corpus.tgt_idx2word[value] for i in range(target_batch.size(0)) for value in target_batch[i].tolist()])

    if bDataRecord:
        #for module_to_hook in model.encoder.layers:
            #handle = module_to_hook.enc_self_attn.register_forward_hook(forward_hook)
        #    handle = module_to_hook.pos_ffn.register_forward_hook(forward_hook)
        #    break
        #handle = model.encoder.src_emb.register_forward_hook(forward_hook)
        #handle = model.encoder.pos_emb.register_forward_hook(forward_hook)
        handle = model.encoder.register_forward_hook(forward_hook)        

    outputs,_,_,_ = model(enc_inputs,dec_inputs)
    if bDataRecord:
        handle.remove()
    #ModelInput[epoch] = copy.deepcopy(forward_hook.inputs)
        for index,values in enumerate(forward_hook.inputs):
            ModelInput[epoch].append(copy.deepcopy(values))
        for index,values in enumerate(forward_hook.outputs):
            ModelOutput[epoch].append(copy.deepcopy(values))
        for index, values in enumerate(model.named_parameters()):
            if(values[1].device.type == 'cuda'):
                model_data = values[1].cpu().data.detach().numpy()
            else:
                model_data = values[1].data.detach().numpy()
            if(epoch == 0):
                Names.append(values[0])
            DataList[index] = copy.deepcopy(model_data)
        Data.append(copy.deepcopy(DataList))
        Epochs.append(epoch)
    loss = criterion(outputs.view(-1,len(corpus.tgt_vocab)),target_batch.view(-1))
    if(epoch + 1) % 1 == 0:
        print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
    loss.backward()
    optimizer.step()

if bDataRecord:
    data = { 'Epochs': Epochs, 'Name': Names, 'Data': Data, 'Input': ModelInput, 'Output': ModelOutput}
    #print(Names)
    pickle.dump(data, file)
    file.close()

os.makedirs(ModelOutDir, exist_ok=True)
CheckPoint = {'model': model,
              'optimizer': optimizer,
             }
torch.save(CheckPoint, os.path.join(ModelOutDir, ModelName))
#方法1
#enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size=1,test_batch=True)
#print(''.join(corpus.src_idx2word[idx.item()] for idx in enc_inputs[0]))
#predict, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
#predict = predict.view(-1,len(corpus.tgt_vocab))
#predict = predict.data.max(1,keepdim=True)[1]
#translated_sentence = [corpus.tgt_idx2word[idx.item()] for idx in predict.squeeze()]
#input_sentence = ''.join(corpus.src_idx2word[idx.item()] for idx in enc_inputs[0])
#print(input_sentence,'->',translated_sentence)

#方法2
enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size=1, device_type = device_type, test_batch = True)
greedy_dec_input = greedy_decoder(model,enc_inputs,start_symbol=corpus.tgt_vocab['<sos>'])
greedy_dec_ooutput_words = [corpus.tgt_idx2word[n.item()] for n in greedy_dec_input.squeeze()]
enc_inputs_words = [corpus.src_idx2word[code.item()] for code in enc_inputs[0]]
print(enc_inputs_words,'->',greedy_dec_ooutput_words)