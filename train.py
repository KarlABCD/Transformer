import torch 
import torch.nn as nn
import torch.optim as optim
from model import Transformer, greedy_decoder, forward_hook
from corpus import TranslationCorpus
import matplotlib.pyplot as plt
import pickle
import copy
import os
import argparse
import importlib.util
from timer import Timer
from animator import Animator
from visualization import ShowHeatmaps

# GPU
def fromfile(filename):
    file_extension = filename.split('.')[-1]
    if file_extension == 'py':
        spec = importlib.util.spec_from_file_location("config_module", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def train(config):

    criterion = nn.CrossEntropyLoss()
    if (os.path.exists(os.path.join(config.ModelOutDir, config.ModelName)) and 
        config.WorkMode == 'PreTrained'):
        PreCheckPt = torch.load(os.path.join(config.ModelOutDir, 
                                        config.ModelName), weights_only=False)
        model = PreCheckPt['model']
        optimizer = PreCheckPt['optimizer']
    else:
        model = Transformer(corpus,config.model_config)
        optimizer = optim.Adam(model.parameters(),lr=config.learningrate)

    if config.device_type == 'cuda' and torch.cuda.is_available():
        model.to('cuda')

    # 数据记录配置
    if config.bDataRecord:
        os.makedirs(config.RecordOutDir, exist_ok=True)
        file = open(os.path.join(config.RecordOutDir, config.RecordName), 'wb')
        Epochs = []
        Names = []
        DataList = [[] for _ in range(len(list(model.named_parameters())))]
        Data = []
        ModelInput =  [[] for _ in range(0,config.epochs)]
        ModelOutput =  [[] for _ in range(0,config.epochs)]
        forward_hook.inputs = []
        forward_hook.outputs = []
        os.makedirs(config.ModelOutDir, exist_ok=True)
    if config.bDebugInfo:
        animator = Animator(xlabel='epoch',
                            ylabel='loss',
                            xlim=[1, config.epochs])
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        enc_inputs, dec_inputs, target_batch = corpus.make_batch(
                                            config.model_config['batch_size'],
                                            config.device_type)
        if config.bDebugInfo:
            for i in range(enc_inputs.size(0)):
                print(f'中文: {corpus.src_vocab.to_tokens(enc_inputs[i])} \n'
                    f'中文token: {enc_inputs[i]} \n'
                    f'英语: {corpus.tgt_vocab.to_tokens(dec_inputs[i])} \n'
                    f'英语token: {dec_inputs[i]}')
        if config.bDataRecord:
            #for module_to_hook in model.encoder.layers:
                #handle = module_to_hook.enc_self_attn.register_forward_hook(forward_hook)
            #    handle = module_to_hook.pos_ffn.register_forward_hook(forward_hook)
            #    break
            #handle = model.encoder.src_emb.register_forward_hook(forward_hook)
            #handle = model.encoder.pos_emb.register_forward_hook(forward_hook)
            handle = model.encoder.register_forward_hook(forward_hook)

        outputs, enc_outputs, dec_outputs, enc_self_attns,\
        dec_self_attns, dec_enc_attns = model(enc_inputs,
                                            dec_inputs, 
                                            config.model_config)
        if config.model_config['bModelDebug']:
            for i in range(config.model_config['n_layers']):
                enc_tick_labels, dec_tick_labels = [], []
                for j in range(enc_inputs.size(0)):
                    enc_tick_labels.append(corpus.src_vocab.
                                           to_tokens(enc_inputs[j].tolist()))
                    dec_tick_labels.append(corpus.tgt_vocab.
                                           to_tokens(dec_inputs[j].tolist()))
                ShowHeatmaps(enc_self_attns[i], xlabel='Enc_Self_Keys', 
                            ylabel='Enc_Self_Queries', 
                            figure_id=1,
                            x_tick_labels=enc_tick_labels,
                            y_tick_labels=enc_tick_labels)
                ShowHeatmaps(dec_self_attns[i], xlabel='Dec_Self_Keys', 
                            ylabel='Dec_Self_Queries', cmap = 'Reds', 
                            figure_id=2,
                            x_tick_labels=dec_tick_labels, 
                            y_tick_labels=dec_tick_labels)
                ShowHeatmaps(dec_enc_attns[i], xlabel='Enc_Dec_Keys', 
                            ylabel='Enc_Dec_Queries',
                            figure_id=3,
                            x_tick_labels=enc_tick_labels,
                            y_tick_labels=dec_tick_labels)
            #print(f'enc_inputs size: {enc_inputs.size()}')
            #print(f'enc_outputs size: {enc_outputs.size()}')
            #print(f'dec_inputs size: {dec_inputs.size()}')
            #print(f'dec_outputs size: {dec_outputs.size()}')
        if config.bDataRecord:
            handle.remove()
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
        loss = criterion(outputs.view(-1,len(corpus.tgt_vocab)),
                         target_batch.view(-1))
        if(epoch + 1) % 1 == 0:
            print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
        loss.backward()
        optimizer.step()
        if (epoch%config.CheckPtNum == 0):
            CheckPoint = {'model': model,
                'optimizer': optimizer,}
            torch.save(CheckPoint, os.path.join(config.ModelOutDir,
                                                config.ModelName))
            print(f'record current number: {epoch}')
        if config.bDebugInfo:
            animator.add(epoch + 1, (float(loss.detach().numpy()), ))

    if config.bDataRecord:
        data = { 'Epochs': Epochs, 'Name': Names, 'Data': Data, 
                 'Input': ModelInput, 'Output': ModelOutput}
        #print(Names)
        pickle.dump(data, file)
        file.close()

    return model, config
def predict(corpus, config, model):
    '''#方法1
    enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size=1,test_batch=True,device_type = config.device_type)
    print(''.join(corpus.src_idx2word[idx.item()] for idx in enc_inputs[0]))
    predict, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    predict = predict.view(-1,len(corpus.tgt_vocab))
    predict = predict.data.max(1,keepdim=True)[1]
    translated_sentence = [corpus.tgt_idx2word[idx.item()] for idx in predict.squeeze()]
    input_sentence = ''.join(corpus.src_idx2word[idx.item()] for idx in enc_inputs[0])
    print(input_sentence,'->',translated_sentence)'''

    #方法2
    '''enc_inputs,_,_ = corpus.make_batch(config.model_config['batch_size'], 
                                       config.device_type,
                                       test_batch = True)'''
    #enc_inputs = torch.tensor([5, 2, 8, 0]).unsqueeze(0)
    enc_inputs = torch.tensor([6, 2, 5, 0]).unsqueeze(0)
    for enc_input in enc_inputs:
        enc_input = torch.unsqueeze(enc_input, dim=0)
        greedy_dec_input = greedy_decoder(model,enc_input, 
                                        tgt_len = corpus.tgt_len, 
                                        start_symbol=corpus.tgt_vocab['<sos>'], 
                                        model_config = config.model_config)
        greedy_dec_ooutput_words = [corpus.tgt_vocab.to_tokens(n.item()) 
                                    for n in greedy_dec_input.squeeze()]
        enc_inputs_words = [corpus.src_vocab.to_tokens(code.item()) 
                            for code in enc_input[0]]
        print(enc_inputs_words,'->',greedy_dec_ooutput_words)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='test config file path')
    args = parser.parse_args()
    config = fromfile(args.config)
    timer = Timer()
    corpus = TranslationCorpus(config.data_dir, config.data_name)
    model, config = train(config)
    timer.stop()
    predict(corpus, config, model)
 #   if config.bDebugInfo:
 #       plt.show()