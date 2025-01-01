import torch
import os
from collections import Counter
from vocab import Vocab
class TranslationCorpus:
    def __init__(self, data_dir, data_name):
        def read_data_nmt():
            with open(os.path.join(data_dir, data_name), 'r', encoding='utf-8') as f:
                return f.read()
        raw_text = read_data_nmt()
        def tokenize_nmt(text, num_examples=None):
            source, target = [], []
            src_len, tgt_len = 0, 0
            for i, line in enumerate(text.split('\n')):
                if num_examples and i > num_examples:
                    break
                parts = line.split('\\')
                if len(parts) == 2:
                    source.append(parts[0].split(' '))
                    if(len(parts[0].split(' ')) > src_len):
                        src_len = len(parts[0].split(' ')) + 1                
                    target.append(parts[1].split(' '))
                    if(len(parts[1].split(' ')) > tgt_len):
                        tgt_len = len(parts[1].split(' ')) + 2
            return source, target, src_len, tgt_len
        self.source, self.target, self.src_len, self.tgt_len = tokenize_nmt(raw_text)
        self.src_vocab = Vocab(self.source, min_freq=0, reserved_tokens=['<pad>'])
        self.tgt_vocab = Vocab(self.target, min_freq=0, reserved_tokens=['<pad>', '<sos>', '<eos>'])
    def make_batch(self, batch_size, device_type, test_batch = False):
        input_batch, output_batch, target_batch = [],[],[]
        if(test_batch):
            #sentence_indices = torch.randperm(len(self.source))[:batch_size]
            sentence_indices = torch.arange(5)
        else:
            sentence_indices = torch.arange(5)
        for index in sentence_indices:
            src_seq = [self.src_vocab[word] for word in self.source[index]]
            tgt_seq = [self.tgt_vocab['<sos>']] + [self.tgt_vocab[word] for word in self.target[index]] + [self.tgt_vocab['<eos>']]
            src_seq += [self.src_vocab['<pad>']] * (self.src_len - len(src_seq))
            tgt_seq += [self.tgt_vocab['<pad>']] * (self.tgt_len - len(tgt_seq))
            input_batch.append(src_seq)
            output_batch.append([self.tgt_vocab['<sos>']] + ([self.tgt_vocab['<pad>']]* \
                                                             (self.tgt_len -2)) if test_batch else tgt_seq[:-1])
            target_batch.append(tgt_seq[1:])
        if device_type == 'cuda':
            input_batch = torch.LongTensor(input_batch).to('cuda')
            output_batch = torch.LongTensor(output_batch).to('cuda')
            target_batch = torch.LongTensor(target_batch).to('cuda') 
        else:
            input_batch = torch.LongTensor(input_batch)
            output_batch = torch.LongTensor(output_batch)
            target_batch = torch.LongTensor(target_batch)
        return input_batch, output_batch, target_batch
    