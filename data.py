import time
import numpy as np
import torchtext
from torchtext import data
import pandas as pd
from random import sample
from utils import tensor2text

class DatasetIterator(object):
    def __init__(self, pos_iter, neg_iter):
        self.pos_iter = pos_iter
        self.neg_iter = neg_iter

    def __iter__(self):
        for batch_pos, batch_neg in zip(iter(self.pos_iter), iter(self.neg_iter)):
            if batch_pos.text.size(0) == batch_neg.text.size(0):#检查是否为text
                #check.text
                yield batch_pos.text, batch_neg.text

def load_dataset(config, train_pos='train.pos', train_neg='train.neg',
                 dev_pos='dev.pos', dev_neg='dev.neg',
                 test_pos='test.pos', test_neg='test.neg'):

    root = config.data_path
    TEXT = data.Field(batch_first=True, eos_token='<eos>')#将batch_size作为第一维度,Field 用于指定指定数据类型如何处理数据进行tensor化
    
    dataset_fn = lambda name: data.TabularDataset(
        path=root + name,
        format='csv',
        fields=[('text', TEXT)]#tuple(str:columns name,Field方法)
    )#创建一个所需数据的数据集
    def sample_dataset(aset,ratio):
        a,b=aset.split(split_ratio=ratio)
        return a
    train_pos_set, train_neg_set = map(dataset_fn, [train_pos, train_neg])
    dev_pos_set, dev_neg_set = map(dataset_fn, [dev_pos, dev_neg])
    test_pos_set, test_neg_set = map(dataset_fn, [test_pos, test_neg])
    if config.sample_ratio < 1:
        train_pos_set, train_neg_set = map(sample_dataset, [train_pos_set, train_neg_set],[config.sample_ratio,config.sample_ratio])
        # dev_pos_set, dev_neg_set = map(sample_dataset, [dev_pos_set, dev_neg_set],[0.05,0.05])
    


    TEXT.build_vocab(train_pos_set, train_neg_set, min_freq=config.min_freq)

    if config.load_pretrained_embed:
        start = time.time()
        
        vectors=torchtext.vocab.GloVe('6B', dim=config.embed_size, cache=config.pretrained_embed_path)
        TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
        print('vectors', TEXT.vocab.vectors.size())
        
        print('load embedding took {:.2f} s.'.format(time.time() - start))

    vocab = TEXT.vocab
        
    dataiter_fn = lambda dataset, train: data.BucketIterator(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=train,
        repeat=train,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        device=config.device
    )#从数据集创建一个数据iterator

    train_pos_iter, train_neg_iter = map(lambda x: dataiter_fn(x, True), [train_pos_set, train_neg_set])
    dev_pos_iter, dev_neg_iter = map(lambda x: dataiter_fn(x, False), [dev_pos_set, dev_neg_set])
    test_pos_iter, test_neg_iter = map(lambda x: dataiter_fn(x, False), [test_pos_set, test_neg_set])

    train_iters = DatasetIterator(train_pos_iter, train_neg_iter)
    dev_iters = DatasetIterator(dev_pos_iter, dev_neg_iter)
    test_iters = DatasetIterator(test_pos_iter, test_neg_iter)
    #check train_pos_iter
    return train_iters, dev_iters, test_iters, vocab
def test_dataset(config, train_pos='train.pos', train_neg='train.neg',
                 dev_pos='dev.pos', dev_neg='dev.neg',
                 test_pos='test.pos', test_neg='test.neg'):

    root = config.data_path
    TEXT = data.Field(batch_first=True, eos_token='<eos>')#将batch_size作为第一维度,Field 用于指定指定数据类型如何处理数据进行tensor化
    
    dataset_fn = lambda name: data.TabularDataset(
        path=root + name,
        format='tsv',
        fields=[('text', TEXT)]#tuple(str:columns name,Field方法)
    )#创建一个所需数据的数据集

    train_pos_set, train_neg_set = map(dataset_fn, [train_pos, train_neg])
    dev_pos_set, dev_neg_set = map(dataset_fn, [dev_pos, dev_neg])
    test_pos_set, test_neg_set = map(dataset_fn, [test_pos, test_neg])

    TEXT.build_vocab(train_pos_set, train_neg_set, min_freq=config.min_freq)

    if config.load_pretrained_embed:
        start = time.time()
        
        vectors=torchtext.vocab.GloVe('6B', dim=config.embed_size, cache=config.pretrained_embed_path)
        TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
        print('vectors', TEXT.vocab.vectors.size())
        
        print('load embedding took {:.2f} s.'.format(time.time() - start))

    vocab = TEXT.vocab
    return train_pos_set

class DropDataset(data.TabularDataset):
    def __init__(self,path,drop_rate,**kwargs):
        examples =[]
        text=[]
        path=root+name
        with open(path,'r') as file:
            for line in file.readlines():
                line = line.strip()
                text.append(line)
        examples = sample(text,drop_rate)
        super(DropDataset,self).__init__(examples,**kwargs)
    
       



    


if __name__ == '__main__':
    import torch
    class Config():
        data_path = './data/yelp/'
        log_dir = 'runs/exp'
        save_path = './save'
        pretrained_embed_path = './embedding/'
        device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
        discriminator_method = 'Multi' # 'Multi' or 'Cond'
        load_pretrained_embed = False
        min_freq = 3
        max_length = 16
        embed_size = 256
        d_model = 256
        h = 4
        num_styles = 2
        num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
        num_layers = 4
        batch_size = 64
        lr_F = 0.0001
        lr_D = 0.0001
        L2 = 0
        iter_D = 10
        iter_F = 5
        F_pretrain_iter = 500
        log_steps = 5
        eval_steps = 25
        learned_pos_embed = True
        dropout = 0
        drop_rate_config = [(1, 0)]
        temperature_config = [(1, 0)]

        slf_factor = 0.25
        cyc_factor = 0.5
        adv_factor = 1

        inp_shuffle_len = 0
        inp_unk_drop_fac = 0
        inp_rand_drop_fac = 0
        inp_drop_prob = 0
    config=Config()
    train_iter, _, _, vocab = load_dataset(config)
    train_pos_set = test_dataset(config)
    # a,b=train_pos_set.split(split_ratio=0.7)
    # print(a,b)
    # for batch in train_iter:
    #     print(batch)
    #     text = tensor2text(vocab, batch[0])
    #     print('\n'.join(text))
    #     break
    ##
   

    
