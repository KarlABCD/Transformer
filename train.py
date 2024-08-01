import torch 
import torch.nn as nn
import torch.optim as optim
from model import Transformer, greedy_decoder
from prepare import TranslationCorpus
from visualization import Visualization

sentences = [
            ['咖哥 喜欢 小冰', 'KaGe likes XiaoBing'],
            ['我 爱 学习 人工智能', 'I love studying AI'],
            ['深度学习 改变 世界', 'DL changed the world'],
            ['自然 语言 处理 很 强大', 'NLP is so powerful'],
            ['神经网络 非常 复杂', 'Neural-Nets are complex']]

corpus = TranslationCorpus(sentences)

model = Transformer(corpus)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    enc_inputs, dec_inputs, target_batch = corpus.make_batch(3)
    outputs,_,_,_ = model(enc_inputs,dec_inputs)
    loss = criterion(outputs.view(-1,len(corpus.tgt_vocab)),target_batch.view(-1))
    if(epoch + 1) % 1 == 0:
        print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
    loss.backward()
    optimizer.step()

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
enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size=1, test_batch = True)
greedy_dec_input = greedy_decoder(model,enc_inputs,start_symbol=corpus.tgt_vocab['<sos>'])
greedy_dec_ooutput_words = [corpus.tgt_idx2word[n.item()] for n in greedy_dec_input.squeeze()]
enc_inputs_words = [corpus.src_idx2word[code.item()] for code in enc_inputs[0]]
print(enc_inputs_words,'->',greedy_dec_ooutput_words)