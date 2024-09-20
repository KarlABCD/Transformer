import torch
from collections import Counter
class TranslationCorpus:
    def __init__(self, sentences):
        self.sentences = sentences
        self.src_len = max(len(sentence[0].split()) for sentence in sentences) + 1
        self.tgt_len = max(len(sentence[1].split()) for sentence in sentences) + 2
        self.src_vocab, self.tgt_vocab = self.create_vocabularies()
        self.src_idx2word = {v: k for k, v in self.src_vocab.items()}
        self.tgt_idx2word = {v: k for k, v in self.tgt_vocab.items()}
    def create_vocabularies(self):
        src_counter = Counter(word for sentence in self.sentences for word in sentence[0].split())
        tgt_counter = Counter(word for sentence in self.sentences for word in sentence[1].split())
        src_vocab = {'<pad>': 0, **{word: i+1 for i, word in enumerate(src_counter)}}
        tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, **{word:i+3 for i, word in enumerate(tgt_counter)}}
        return src_vocab, tgt_vocab
    def make_batch(self, batch_size, device_type, test_batch = False):
        input_batch, output_batch, target_batch = [],[],[]
        sentence_indices = torch.randperm(len(self.sentences))[:batch_size]
        for index in sentence_indices:
            src_sentence, tgt_sentence = self.sentences[index]
            src_seq = [self.src_vocab[word] for word in src_sentence.split()]
            tgt_seq = [self.tgt_vocab['<sos>']] + [self.tgt_vocab[word] for word in tgt_sentence.split()] + [self.tgt_vocab['<eos>']]
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
    