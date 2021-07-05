import transformers
from pl_en import prepare_corpus, EN_PL_parallel

def main():
    model = transformers.AutoModelWithLMHead.from_pretrained('t5-small')
    prepare_corpus('pl-en', 400)
    window = 3
    dim = 64
    en_pl = EN_PL_parallel('.', window,1000, embedding_dim=dim)
    # model = nn.Transformer(dim, 2, 2, 2, dim*4, batch_first=False)
    vocab_size = en_pl.from_ds.bpe.vocab_size()
    # model = LanguageTransformer(vocab_size, dim, 2, 2,
                                # 2, dim*4, 50,
                                # 0., 0.)
if __name__ == '__main__':
    main()