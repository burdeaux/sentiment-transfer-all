import torch
import os
from data import load_dataset
def write_vocab(vocab,path):
    #<eos> token will change to <s> and <s/> for other models
    with open(path, 'w') as f:     
        for token, index in vocab.stoi.items():
            f.write(f'{token}\n')

class Config():
    sample_ratio=1.0
    data_path="data/yelp_20_2/"
    min_freq = 2
    load_pretrained_embed = False
    batch_size = 32
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')


def main():
    config=Config()
    train_iters, dev_iters, test_iters, vocab = load_dataset(config)
    print('Vocab size:',len(vocab))
    write_file=os.path.join(config.data_path,'vocab.txt')
    write_vocab(vocab,write_file)
    
if __name__ == '__main__':
    main()
