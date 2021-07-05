import numpy as np
import os
from os.path import exists
import xml.etree.ElementTree as ET
from lxml import etree
from numpy.core.fromnumeric import squeeze
from torch.nn.modules.loss import CrossEntropyLoss
from common import my_random_split, change_cwd, Timer

import torchtext as ttxt
import random

import youtokentome as yttm
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader
import torch
import sys

def tokenize_test(train_data_path="subtitles_ru.txt", model_path="example.model"):

    # Training model
    yttm.BPE.train(data=train_data_path, vocab_size=30000, model=model_path)

    # Loading model
    bpe = yttm.BPE(model=model_path)

    test_text = ['купила мама Лёше отличные калоши.', 
    'Врач сказал, что ограничительные меры достаточно логичны и что по-другому в сложившейся ситуации государство действовать не может. По словам Лунченкова, чем раньше население начнет следовать этим мерам и прививаться, тем быстрее сформируется коллективный иммунитет. Инфекционист сказал, что необходимость ревакцинации зависит от вакцины и технологии, по которой она была изготовлена. Лунченков заявил, что дети находятся в последней категории граждан для получения вакцины, потому что для них вирус наименее опасен. Тем не менее, по словам врача, они могут являться переносчиками ковида и поэтому представлять опасность для более взрослых членов семьи.'
    ]
    # Two types of tokenization
    print(bpe.encode([test_text[1]], output_type=yttm.OutputType.ID))
    print(bpe.encode([test_text[1]], output_type=yttm.OutputType.SUBWORD))

def read_xml_corpus(path, use_lxml=True):
    corpus_list = []
    timer = Timer()
    last_id = 0
    iter_xml = ET.iterparse(path)
    if use_lxml:
        iter_xml = etree.iterparse(path)
    for i, (_, sent) in enumerate(iter_xml):
        if sent.tag != 's':
            continue
        if '\n' in sent.text:
            print(f'~~~~~~~~~~!!!!!!!!!!!!!!!!1111 {path}')
        corpus_list.append(sent.text)

        id = int(sent.attrib['id'])
        if id != last_id + 1:
            print(id, corpus_list[-1])
        sent.clear()
        last_id = id

    timer.get(f'parse {len(corpus_list)} sentences')
    return corpus_list

class PL_RU_parallel:
    def __init__(self, path, window):
        path = Path(path)
        from_corpus_txt = "from.txt"
        to_corpus_txt = 'to.txt'
        
        root = ET.parse(path).getroot()
        link_group = root.find('linkGrp')
        from_to_text = list(link_group.attrib.values())[1:]

        self.link = []
        for i, link in enumerate(link_group.findall('link')):
            lines_ref = link.attrib['xtargets'].split(';')
            from_lines = [int(num) for num in lines_ref[0].split(' ')]
            to_lines = [int(num) for num in lines_ref[1].split(' ')]
            self.link.append([from_lines, to_lines])
        if not (exists(from_corpus_txt) and exists(from_corpus_txt)):
            self.from_list = read_xml_corpus(from_to_text[0], False)
            self.to_list = read_xml_corpus(from_to_text[1], False)

            self.corpus_to_text_files(from_corpus_txt, to_corpus_txt)
            del self.from_list
            del self.corpus_to_text_files


        from_bpe_model = 'from_bpe.model'
        to_bpe_model = 'to_bpe.model'
        
        self.from_ds = BPE_ds(from_corpus_txt, from_bpe_model, window)
        self.to_ds = BPE_ds(to_corpus_txt, to_bpe_model, window)
        self.corpus_from_text_files(from_corpus_txt, to_corpus_txt)
        self.len = -1

    def corpus_to_text_files(self, from_filename='from.txt', to_filename='to.txt'):
        with open(from_filename, 'w') as f:
            print(*self.from_list, sep='\n', file=f)
        with open(to_filename, 'w') as f:
            print(*self.to_list, sep='\n', file=f)

    def corpus_from_text_files(self, from_filename='from.txt', to_filename='to.txt'):
        with open(from_filename, 'r') as f:
            self.from_list = [line.strip() for line in f]
        with open(to_filename, 'r') as f:
            self.to_list = [line.strip() for line in f]

    def __len__(self):
        return self.len
    
    def __getitem__(self):
        pass

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_size=64, window_size=3, hidden_size=64):
        super().__init__()
        self.relu = nn.ReLU()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.embeds_to_hidden = nn.Sequential(nn.Linear(2*window_size*embed_size, hidden_size), self.relu)
        self.hidden_to_vocab = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, target=None):
        batch_size = inputs.shape[0]
        embeds = self.embeddings(inputs)
        hid = self.embeds_to_hidden(embeds.view(batch_size, -1))
        if target is None:
            out = self.hidden_to_vocab(hid)
            return out
        else:
            positive_loss = (hid.unsqueeze(1) @ self.hidden_to_vocab.weight[target].unsqueeze(2)).squeeze().sigmoid().log
            negative_loss = torch.linspace()
            print(negative_loss.shape)
            assert False
            positive = hid @ self.hidden_to_vocab.weight[0]
            assert False, f'{hid.shape}x{self.hidden_to_vocab.weight.shape}={out.shape} .. {positive.shape}'
        # out = hid @ self.embeddings.weight.T
        # assert False, f'{hid.shape}x{self.hidden_to_vocab.weight.shape}={out.shape}'
        return out
    
class BPE_ds:
    def __init__(self, train_data_path, bpe_path, window, vocab_size=10000, coverage=0.99):
        # super().__init__()
        timer = Timer()
        yttm.BPE.train(data=train_data_path, model=bpe_path, vocab_size=vocab_size, coverage=coverage,)
        self.bpe = yttm.BPE(bpe_path)
        timer.get('bpe generated')
        sentences = []
        # return
        with open(train_data_path + '.bpe', 'w') as f_bpe:
            with open(train_data_path, 'r') as f:
                # sentences = [self.bpe.encode([line.strip()], output_type=yttm.OutputType.ID) for line in f]
                for line in f:
                    sentences.append(self.bpe.encode([line.strip()], output_type=yttm.OutputType.ID, bos=True, eos=True)[0])
                    # print(*self.bpe.encode([line.strip()], output_type=yttm.OutputType.ID)[0], file=f_bpe)

        timer.get(f'readed {len(sentences)} sentences')
        use_dtype = torch.long
        if vocab_size >= (2 ** 15 - 1):
            use_dtype = torch.int32

        self.context = []#torch.empty((0, 2*window), dtype=use_dtype)
        self.center_word = []#torch.empty((0,), dtype=use_dtype)
        self.len = 0
        for sent in sentences:
            self.len += max(0, len(sent) - 2*window)
            # if len(sent) - 2*window > 0:
                # print(sent)
            for i, center_word in enumerate(sent[window:-window]):
                # self.center_word = torch.cat((self.center_word, torch.tensor([center_word], dtype=use_dtype)), dim=0)
                self.center_word.append(center_word)
                # context = torch.tensor(sent[i:i + window] + sent[i + window + 1:i + 2*window + 1], dtype=use_dtype).view(1, 2*window)
                # self.context = torch.cat((self.context, context), dim=0)
                self.context.append(sent[i:i + window] + sent[i + window + 1:i + 2*window + 1])
                # self.data.append([sent[i:i + window] + sent[i + window + 1:i + 2*window + 1], center_word])
            # if self.len % 1e4 == 0:
                # print('.', end='', flush=True)
        del sentences
        self.context = torch.tensor(self.context, dtype=use_dtype)
        self.center_word = torch.tensor(self.center_word, dtype=use_dtype)
        timer.get(f'gen {self.len} data.')
        print(self.center_word.shape, self.context.shape)
    
    def encode(self, text_list):
        return self.bpe.encode(text_list, yttm.OutputType.ID, True, True)

    def decode(self, sentence_ids_list):
        return [self.bpe.decode(sent, ignore_ids=[0,1,2,3]) for sent in sentence_ids_list] 


    def __getitem__(self, i):
        return self.context[i], self.center_word[i]

    def __len__(self):
        return self.len

def loss2(pred, target, embeds):
    if True:
        with torch.no_grad():
            mx = pred.argmax(dim=-1)
        mtrx = embeds[mx]
        # print(f'{mtrx.shape} {embeds[target].shape}')
        pos_loss = (mtrx.unsqueeze(1) @ embeds[target].unsqueeze(2)).squeeze().log().mean()
    if False:
        noise_emb = torch.randint(0,embeds.shape[0], (target.shape[0], 400), dtype=torch.long)
        noise_emb = noise_emb[noise_emb != target]
        noise_emb = embeds[noise_emb]
        neg_loss = (noise_emb @ embeds[target].unsqueeze(2)).squeeze().sigmoid().log().mean()
    return pos_loss
    
def train_cbow(cbow, bpe_ds, valid=None, epoch_number=2):
    if valid is None:
        train, test, valid = my_random_split(bpe_ds, [87, 8, 5])
    else:
        train, test = my_random_split(bpe_ds, [9, 1])

    

    lsm = nn.LogSoftmax(dim=-1)
    lossFunc = nn.NLLLoss()
    CE = CrossEntropyLoss()
    #############################
    # Добавить лосс: максимизировать таргет-ембеддинг * k max-out
    #############################
    timer = Timer()
    optim = torch.optim.Adam(cbow.parameters(), lr=1e-3, weight_decay=1e-3)
    R = 1
    E = epoch_number // R
    max_bs = 256
    for epoch in range(E):
        av_loss = []
        # for i, (batch_context, batch_center) in enumerate(DataLoader(train, int(max_bs / E * (1 + epoch)), False)):
        for i, (batch_context, batch_center) in enumerate(DataLoader(train, 128, True)):
            batch_context = batch_context.long()
            batch_center = batch_center.long()
            for r in range(R):
                cbow.zero_grad()
                prob_center = cbow(batch_context)#, batch_center)
                if True or epoch < 3:
                    loss = CE(prob_center, batch_center)#lossFunc(lsm(prob_center), batch_center)
                else:
                    loss = CE(prob_center, batch_center) + loss2(prob_center, batch_center, cbow.hidden_to_vocab.weight)
                loss.backward()
                optim.step()

            av_loss.append(loss.item())
        if True or epoch % 5 == 0:
            cbow.eval()
            with torch.no_grad():
                correct = 0
                all = 0
                for i, (batch_context, batch_center) in enumerate(DataLoader(test, 64, False)):
                    batch_context = batch_context.long()
                    batch_center = batch_center.long()
                    pred_centers = cbow(batch_context).argmax(dim=-1)
                    correct += sum(pred_centers == batch_center)
                    all += batch_center.shape[0]
            timer.get(f'{epoch: 4} av_loss = {np.mean(av_loss):.3e}, acc = {correct/all:.4f}')
            cbow.train()
    if True:
        cbow.eval()
        with torch.no_grad():
            correct = 0
            all = 0
            for i, (batch_context, batch_center) in enumerate(DataLoader(valid, 64, False)):
                batch_context = batch_context.long()
                batch_center = batch_center.long()
                pred_centers = cbow(batch_context).argmax(dim=-1)
                correct += sum(pred_centers == batch_center)
                all += batch_center.shape[0]
        timer.get(f'                    valid_acc = {correct/all:.4f}')
        cbow.train()

# from gensim import models
import gensim

def word2vec(train_ru_data_path="subtitles_ru.txt", ru_vocab_size = 10000):
    ru_bpe_path = 'ru_bpe.model'
    
    cbow_window = 4
    
    bpe_ru_ds = BPE_ds(train_ru_data_path, ru_bpe_path, vocab_size=ru_vocab_size, window=cbow_window)
    cbow_from = CBOW(bpe_ru_ds.bpe.vocab_size(), window_size=cbow_window)

    print(bpe_ru_ds.bpe.vocab_size())
    print(cbow_from.embeddings.weight.shape, cbow_from.hidden_to_vocab.weight.shape)
    # return
    ds, valid = my_random_split(bpe_ru_ds, [95,5])
    train_cbow(cbow_from, ds, epoch_number=3, valid=valid)
    train_cbow(cbow_from, ds, epoch_number=3, valid=valid)
    train_cbow(cbow_from, ds, epoch_number=3, valid=valid)
    train_cbow(cbow_from, ds, epoch_number=3, valid=valid)


    # cbow_from_gensim = gensim.models.Word2Vec(ds.dataset)
    return
    print(f'epochs done with ')
    test_text = ['купила мама Лёше отличные калоши.', 
    'Врач сказал, что ограничительные меры достаточно логичны и что по-другому в сложившейся ситуации государство действовать не может. По словам Лунченкова, чем раньше население начнет следовать этим мерам и прививаться, тем быстрее сформируется коллективный иммунитет. Инфекционист сказал, что необходимость ревакцинации зависит от вакцины и технологии, по которой она была изготовлена. Лунченков заявил, что дети находятся в последней категории граждан для получения вакцины, потому что для них вирус наименее опасен. Тем не менее, по словам врача, они могут являться переносчиками ковида и поэтому представлять опасность для более взрослых членов семьи.'
    ]
    # Two types of tokenization
    # bpe_1sent = bpe_ru_ds.bpe.encode([test_text[1]], output_type=yttm.OutputType.ID)
    # print(type(bpe_1sent[0]))
    # print(bpe_ru_ds.bpe.encode([test_text[1]], output_type=yttm.OutputType.SUBWORD))

def download_corpus():
    pass

def main():
    word2vec('test.ru', 10000)
    return
    download_corpus()
    window = 2
    pl_ru = PL_RU_parallel('pl-ru.xml', window)
    model = CBOW(10000, window )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weigth_decay=0.)
    LossFunc = nn.NLLLoss()
    log_softmax = nn.LogSigmoid(dim=-1)
    for (batch_x, batch_y) in DataLoader(pl_ru.to_ds, 1, False):
        model.zero_grad()
        center_words_logits = model(batch_x)
        loss = LossFunc(log_softmax(center_words_logits), batch_y)
        loss.backward()
        optim.step()
        print(loss.item())
    return

if __name__ == '__main__':
    change_cwd(__file__)
    if False:
        v = torch.randint(0,10,(10,2), dtype=torch.long,)
        
        w = torch.randint(0,10,(10,1), dtype=torch.long,)
        
        print(v)
        print(w)
        print(v[v != w].shape)
        assert False
    main()

