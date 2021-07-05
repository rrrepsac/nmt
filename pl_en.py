from os.path import exists
from pathlib import Path
import os

from torch.utils.data.dataloader import DataLoader
from common import timing, Timer, change_cwd

from tokenizer import BPE_ds, CBOW, train_cbow
import torch
from torch import nn
from collections import Counter
from statistics import quantiles
import math

from model import LanguageTransformer


def prepare_corpus(root_path, max_lines=-1):
    files = []
    timer = Timer()
    for root, dirs, cur_files in os.walk(root_path):
        if 'europar' in cur_files[0]:
            files = [Path(root)/file for file in cur_files]
            break
    # print(files)
    lines_number = 0
    cn = 0
    all = 0
    sentences_en = []
    sentences_pl = []
    with open(files[0], 'r', encoding='utf-8') as f_en:
        with open(files[1], 'r', encoding='utf-8') as f_pl:
            while True:
                line_en = f_en.readline()
                line_pl = f_pl.readline()
                if line_en == '' or line_en == '':
                    timer.get(f'{lines_number}, {cn} {line_en} {line_pl}')
                    break
                line_en = line_en.strip()
                line_pl = line_pl.strip()
                if line_en.endswith('('):
                    line_en = line_en[:-1]
                if line_pl.endswith('('):
                    line_pl = line_pl[:-1]
                all += 1
                if line_en == line_pl:
                    continue
                if len(line_en) < 10 or len(line_pl) < 10:
                    continue
                p = len(line_pl) / len(line_en)
                if p < 1:
                    p = 1/p
                if p > 2:
                    continue
                    # print(all, lines_number, line_en.strip(), f'len={len(line_en.strip())}', '__', line_pl.strip(), f'len = {len(line_pl.strip())}', '<<')
                    # cn += 1
                sentences_en.append(line_en)
                sentences_pl.append(line_pl)
                lines_number += 1
                if max_lines > 0 and lines_number > max_lines:
                    break
    with open('norm.en', 'w') as fn_en:
        print(*sentences_en, sep='\n', file=fn_en)
    with open('norm.pl', 'w') as fn_pl:
        print(*sentences_pl, sep='\n', file=fn_pl)
    return

def sort_parallel_corpus(from_list, to_list):
    sorted_positions = sorted(range(len(from_list)), key=lambda i: len(from_list[i]) + len(to_list[i]))
    coinc_pos = 0
    for i, (c,n) in enumerate(zip(sorted_positions[:-1], sorted_positions[1:])):
        if from_list[c] == from_list[n] and to_list[c] == to_list[n]:
            sorted_positions.pop(i - coinc_pos)
            coinc_pos += 1
    print('coinc: ', coinc_pos)

    return [from_list[i] for i in sorted_positions], [to_list  [i] for i in sorted_positions]

    
class EN_PL_parallel:
    def __init__(self, path, window, vocab_size=10000, embedding_dim=64, train_embed=True):
        path = Path(path)
        self.vocab_dtype = torch.int16
        if vocab_size > (2**15 - 1):
            self.vocab_dtype = torch.int32
        from_corpus_txt = "norm.en"
        to_corpus_txt   = 'norm.pl'
        
        from_bpe_model = 'from_bpe.model'
        to_bpe_model   = 'to_bpe.model'

        self.from_ds = BPE_ds(from_corpus_txt, from_bpe_model, window, vocab_size)
        self.cbow_from = CBOW(self.from_ds.bpe.vocab_size(), window_size=window,
        embed_size=embedding_dim, hidden_size=embedding_dim)
        if train_embed:
            timing(train_cbow)(self.cbow_from, self.from_ds, epoch_number=3)

        self.to_ds = BPE_ds(to_corpus_txt, to_bpe_model, window, vocab_size)
        self.cbow_to = CBOW(self.to_ds.bpe.vocab_size(), window_size=window,
        embed_size=embedding_dim, hidden_size=embedding_dim)
        if train_embed:
            timing(train_cbow)(self.cbow_to, self.to_ds, epoch_number=3)

        self.from_list = {'train' : [], 'test' : [], 'valid' : []}
        self.to_list = {'train' : [], 'test' : [], 'valid' : []}

        self.corpus_from_text_files(from_corpus_txt, to_corpus_txt)

        for key in self.from_list.keys():
            assert len(self.from_list[key]) == len(self.to_list[key]), f'length of from[{key}] to[{key}] list is different'
        self.len = len(self.from_list)

    def gen_batch(self, src, max_src_seq_len, tgt, max_tgt_seq_len):
        for seq in src:
            seq += [0]*(max_src_seq_len - len(seq))
        for seq in tgt:
            seq += [0]*(max_tgt_seq_len - len(seq))
        src = torch.tensor(src)
        tgt = torch.tensor(tgt)
        return src, tgt

    def parallel_data_loader(self, key, batch_volume):  # batch_volume = (max_src_seq_len + max_tgt_seq_len) * batch_size
        src_list = []
        tgt_list = []
        width = 0
        batch_size = 0
        max_src_seq_len = 0
        max_tgt_seq_len = 0
        for i, (src_seq, tgt_seq) in enumerate(zip(self.from_list[key], self.to_list[key])):
            src_list.append(src_seq)
            tgt_list.append(tgt_seq)
            batch_size += 1
            max_src_seq_len = max(max_src_seq_len, len(src_seq))
            max_tgt_seq_len = max(max_tgt_seq_len, len(tgt_seq))
            width = max_src_seq_len + max_tgt_seq_len
            if width * (batch_size + 1) > batch_volume or i == len(self.from_list[key]) - 1:
                yield self.gen_batch(src_list, max_src_seq_len, tgt_list, max_tgt_seq_len)
                src_list.clear()
                tgt_list.clear()
                batch_size = 0




    def corpus_to_text_files(self, from_filename='from.txt', to_filename='to.txt'):
        with open(from_filename, 'w') as f:
            print(*self.from_list, sep='\n', file=f)
        with open(to_filename, 'w') as f:
            print(*self.to_list, sep='\n', file=f)

    def corpus_from_text_files(self, from_filename='from.txt', to_filename='to.txt'):
        timer = Timer()
        from_list = []
        to_list = []

        with open(from_filename, 'r') as f:
            from_list = [line.strip() for line in f]
        from_list = self.from_ds.encode(from_list)
        timer.get(f'{quantiles([len(s) for s in from_list], n=40)}')

        with open(to_filename, 'r') as f:
            to_list = [line.strip() for line in f]
        to_list = self.to_ds.encode(to_list)
        timer.get(f'{quantiles([len(s) for s in to_list], n=40)}')

        max_seq_len = 50
        good_list = [i for i, s in enumerate(from_list) if len(s) < max_seq_len and len(to_list[i]) < max_seq_len - 2]
        from_list = [from_list[i] for i in good_list]
        to_list   = [to_list  [i] for i in good_list]
        timer.get()

        ratios = [.85, .10, .5]
        pos = 0
        for r, key in zip(ratios, self.from_list.keys()):
            new_pos = pos + int(r*len(from_list))
            if new_pos >= len(from_list) - 3:
                new_pos = len(from_list)
            self.from_list[key] = from_list[pos:new_pos]
            self.to_list  [key] = to_list  [pos:new_pos]
            self.from_list[key], self.to_list[key] = sort_parallel_corpus(self.from_list[key], self.to_list[key])
            pos = new_pos
        
        # to_len = Counter([len(s) for s in self.to_list])
        # print(to_len)
        timer.get()

    def __len__(self):
        return self.len
    
    def __getitem__(self):
        pass

class LT(nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, inp, tgt,):
        pass

def main():
    prepare_corpus('pl-en', 40000)
    window = 3
    dim = 64
    en_pl = EN_PL_parallel('.', window,10000, embedding_dim=dim, train_embed=False)
    # model = nn.Transformer(dim, 2, 2, 2, dim*4, batch_first=False)
    vocab_size = en_pl.from_ds.bpe.vocab_size()
    model = LanguageTransformer(vocab_size, dim, 2, 2,
                                2, dim*4, 50,
                                0., 0.)

    # Use Xavier normal initialization in the transformer
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    loss_f = nn.CrossEntropyLoss()#ignore_index=0)
    # optim = torch.optim.Adam(list(model.parameters() + list(en_pl.cbow_from.embeddings.parameters()) + \
    # list(en_pl.cbow_to.embeddings.parameters()) + list(en_pl.cbow_to.hidden_to_vocab.parameters()
    # ), lr=1e-3, weight_decay=0.)
    optim = torch.optim.Adam(list(model.parameters()), lr=1e-3, weight_decay=0.)
    batch_first = True
    for e in range(10):
        for i, (x, y) in enumerate(en_pl.parallel_data_loader('train', batch_volume=10000)):
            if not batch_first:
                x = x.permute(1,0)
                y = y.permute(1,0)
            # print(i, x.shape, y.shape)
            # en_pl.cbow_from.requires_grad_(True)
            # en_pl.cbow_to.requires_grad_(True)
            # inp = en_pl.cbow_from.embeddings(x).permute(1,0,2)
            # tgt = en_pl.cbow_to.embeddings(y[:,:-1]).permute(1,0,2)
            inp = x
            tgt = y[:, :-1]
            inp_key_padding_mask = (x == 0)
            mem_key_padding_mask = inp_key_padding_mask
            tgt_key_padding_mask = (y[:,:-1] == 0)
            tgt_mask = model.transformer.generate_square_subsequent_mask(tgt.shape[int(batch_first)])
            labels = y[:, 1:]
            logits = model(inp, tgt,
                tgt_mask=tgt_mask,
                memory_key_padding_mask = mem_key_padding_mask,
                src_key_padding_mask=inp_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            pred = logits#en_pl.cbow_to.hidden_to_vocab(logits)
            # print(inp.shape, tgt.shape, pred.shape, labels.shape)
            assert (y[:,1:] == labels).all(), 'copy error occured\n'
            # loss = loss_f(pred.view(-1, en_pl.to_ds.bpe.vocab_size()), labels.reshape(-1))
            gt = []
            pred_red = []
            loss = 0
            q = 0
            for p, g in zip(pred.view(-1, en_pl.to_ds.bpe.vocab_size()), labels.reshape(-1)):
                if g != 0:
                    loss += loss_f(p.view(1,-1), g.view(1))
                    q += 1
            loss /= q
            loss.backward()
            if i % 20 == 0:
                print(f'{i} loss= {loss.item()}')
            if i == 0:
                num = torch.randint(0, y.shape[int(batch_first)] - 1, (1,))[0]
                pred_0 = pred.argmax(dim=-1)[:,num].long()
                print(pred_0.tolist())
                print(en_pl.to_ds.decode([pred_0.tolist()]))
                # print(en_pl.to_ds.decode([y[num, :].tolist()]))
                subw = [en_pl.to_ds.bpe.id_to_subword(x) for x in y[num, :].tolist() if x != 0]
                print(subw)
            optim.step()
    return

if __name__ == '__main__':
    change_cwd(__file__)
    print(torch.__version__)
    main()