from models.transformer import NatureGAN
import torch
import time
from data import load_dataset
from models import StyleTransformer, Discriminator
from train import train, auto_eval
from train_lowtem import train_low


class Config():
    data_path = './data/yelp_20_2/'
    log_dir = 'runs/exp'
    save_path = './save'
    pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Cond' # 'Multi' or 'Cond'
    load_pretrained_embed = False
    min_freq = 3
    tolerance=3
    max_input_length=30
    max_length = 32
    embed_size = 256
    d_model = 256
    h = 4
    num_styles = 2
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    num_layers = 4
    batch_size = 32
    lr_F = 0.0001
    lr_D = 0.0001
    lr_N = 0.0001
    L2 = 0
    iter_D = 10
    iter_F = 5
    iter_N = 1
    F_pretrain_iter = 500
    log_steps = 25
    eval_steps = 50
    learned_pos_embed = True
    dropout = 0 
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]

    slf_factor = 0.25
    cyc_factor = 1
    adv_factor = 1
    n_adv_factor = 1

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0
    exp_num = 7_20

    sample_ratio = 1.0


def main():
    config = Config()
    train_iters, dev_iters, test_iters, vocab = load_dataset(config,train_pos='train.pos.csv', train_neg='train.neg',
                 dev_pos='dev.pos.csv', dev_neg='dev.neg.csv',
                 test_pos='test.pos.csv', test_neg='test.neg.csv')
    print('Vocab size:', len(vocab))
    model_F = StyleTransformer(config, vocab).to(config.device)
    model_D = Discriminator(config, vocab).to(config.device)
    model_N = NatureGAN(config,vocab).to(config.device)
    print(config.discriminator_method)
    
    train(config, vocab, model_F, model_D,model_N, train_iters, dev_iters, test_iters)
    

if __name__ == '__main__':
    main()
