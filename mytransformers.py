from numpy.lib import math
import torch
from torch import nn
import math
from common import Timer

class transformers(nn.Module):
    def __init__(self, d_model, nhead, num_layers, encoder_embeddings_weights, decoder_embeddings_weights,\
        norm_layer=nn.LayerNorm, max_len=5000):
        super().__init__()
        emb_freezing = True        
        self.encoder_embeddings = nn.Embedding(encoder_embeddings_weights.shape[0], d_model).from_pretrained(encoder_embeddings_weights, freeze=emb_freezing)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers, norm_layer)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.decoder  = nn.TransformerDecoder(decoder_layer, num_layers, norm_layer)
        self.decoder_embeddings = nn.Embedding(decoder_embeddings_weights.shape[0], d_model).from_pretrained(decoder_embeddings_weights, freeze=emb_freezing)
        self.decoder_linear = nn.Linear(d_model, decoder_embeddings_weights.shape[0])

    def forward(self, inputs, targets=None, teacher_forcing=0.5):
        embs = self.encoder_embeddings(inputs) + self.pe[:inputs.shape[0], :]  # ?? seq_len
        memory = self.encoder(embs)
        max_outputs_len = 1000
        if targets:
            max_outputs_len = targets.shape[0]
        if targets:
            target_embs = 
        tgt = self.decoder_embeddings(targets[:-1]) + self.pe[:targets.shape[0] - 1, :] # <bos>
        size = max_outputs_len
        tgt_mask = torch.triu(torch.ones((size, size)))
        outputs = self.decoder(tgt, memory, tgt_mask)
        return outputs

def main():
    tr = transformers(64, 8, 3)
    dl = None
    loss_f = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(tr.parameters(), lr=1e-3, weight_decay=0e-3)
    timer = Timer()
    for i, (batch_inputs, batch_targets) in dl:
        tr.zero_grad()
        logits = tr(batch_inputs, batch_targets)
        loss = loss_f(logits.view(-1), batch_targets.view(-1))
        loss.backward()
        optim.step()
        timer.get(f'loss = {loss:.3e}')
    return


if __name__ == '__main__':
    main()