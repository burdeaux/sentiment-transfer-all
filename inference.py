import os
import time
import torch
import numpy as np
from torch import nn, optim
#from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from evaluator import Evaluator
from utils import tensor2text, calc_ppl, idx2onehot, add_noise, word_drop
from models.transformer import NatureGAN
from data import load_dataset
from models import StyleTransformer, Discriminator
from train import train, auto_eval
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
    lr_N = 0.0001
    L2 = 0
    iter_D = 10
    iter_F = 5
    iter_N = 5
    F_pretrain_iter = 500
    log_steps = 25
    eval_steps = 50
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]

    slf_factor = 0.25
    cyc_factor = 0.5
    adv_factor = 1
    n_adv_factor = 1

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0
def get_lengths(tokens, eos_idx):
    lengths = torch.cumsum(tokens == eos_idx, 1)
    lengths = (lengths == 0).long().sum(-1)
    lengths = lengths + 1 # +1 for <eos> token
    return lengths
def auto_eval(config, vocab, model_F, test_iters, global_step, temperature):
    model_F.eval()
    vocab_size = len(vocab)
    eos_idx = vocab.stoi['<eos>']

    def inference(data_iter, raw_style):
        gold_text = []
        raw_output = []
        rev_output = []
        for batch in data_iter:
            inp_tokens = batch.text
            inp_lengths = get_lengths(inp_tokens, eos_idx)
            raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
            rev_styles = 1 - raw_styles
        
            with torch.no_grad():
                raw_log_probs = model_F(
                    inp_tokens,
                    None,
                    inp_lengths,
                    raw_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )
            
            with torch.no_grad():
                rev_log_probs = model_F(
                    inp_tokens, 
                    None,
                    inp_lengths,
                    rev_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )
                
            gold_text += tensor2text(vocab, inp_tokens.cpu())
            raw_output += tensor2text(vocab, raw_log_probs.argmax(-1).cpu())
            rev_output += tensor2text(vocab, rev_log_probs.argmax(-1).cpu())

        return gold_text, raw_output, rev_output

    pos_iter = test_iters.pos_iter
    neg_iter = test_iters.neg_iter
    
    gold_text, raw_output, rev_output = zip(inference(neg_iter, 0), inference(pos_iter, 1))


    evaluator = Evaluator()
    ref_text = evaluator.yelp_ref

    
    acc_neg = evaluator.yelp_acc_0(rev_output[0])
    acc_pos = evaluator.yelp_acc_1(rev_output[1])
    bleu_neg = evaluator.yelp_ref_bleu_0(rev_output[0])
    bleu_pos = evaluator.yelp_ref_bleu_1(rev_output[1])
    # ppl_neg = evaluator.yelp_ppl(rev_output[0])
    # ppl_pos = evaluator.yelp_ppl(rev_output[1])

    for k in range(5):
        idx = np.random.randint(len(rev_output[0]))
        print('*' * 20, 'neg sample', '*' * 20)
        print('[gold]', gold_text[0][idx])
        print('[raw ]', raw_output[0][idx])
        print('[rev ]', rev_output[0][idx])
        print('[ref ]', ref_text[0][idx])

    print('*' * 20, '********', '*' * 20)
    

    for k in range(5):
        idx = np.random.randint(len(rev_output[1]))
        print('*' * 20, 'pos sample', '*' * 20)
        print('[gold]', gold_text[1][idx])
        print('[raw ]', raw_output[1][idx])
        print('[rev ]', rev_output[1][idx])
        print('[ref ]', ref_text[1][idx])

    print('*' * 20, '********', '*' * 20)

#不打印ppl设为0
    print(('[auto_eval] acc_pos: {:.4f} acc_neg: {:.4f} ' + \
          'bleu_pos: {:.4f} bleu_neg: {:.4f} ' + \
          'ppl_pos: {:.4f} ppl_neg: {:.4f}\n').format(
              acc_pos, acc_neg, bleu_pos, bleu_neg, 0, 0,
    ))

    
    # save output
    # save_file = config.save_folder + '/' + str(global_step) + '.txt'
    # eval_log_file = config.save_folder + '/eval_log.txt'
    save_file_ppl_0=config.save_folder+'/'+'ppl_0'+str(global_step)+'.txt'
    save_file_ppl_1=config.save_folder+'/'+'ppl_1'+str(global_step)+'.txt'
    # with open(eval_log_file, 'a') as fl:
    #     print(('iter{:5d}:  acc_pos: {:.4f} acc_neg: {:.4f} ' + \
    #            'bleu_pos: {:.4f} bleu_neg: {:.4f} ' + \
    #            'ppl_pos: {:.4f} ppl_neg: {:.4f}\n').format(
    #         global_step, acc_pos, acc_neg, bleu_pos, bleu_neg, 0, 0,
    #     ), file=fl)
    # with open(save_file, 'w') as fw:
    #     print(('[auto_eval] acc_pos: {:.4f} acc_neg: {:.4f} ' + \
    #            'bleu_pos: {:.4f} bleu_neg: {:.4f} ' + \
    #            'ppl_pos: {:.4f} ppl_neg: {:.4f}\n').format(
    #         acc_pos, acc_neg, bleu_pos, bleu_neg, 0, 0,
    #     ), file=fw)

    #     for idx in range(len(rev_output[0])):
    #         print('*' * 20, 'neg sample', '*' * 20, file=fw)
    #         print('[gold]', gold_text[0][idx], file=fw)
    #         print('[raw ]', raw_output[0][idx], file=fw)
    #         print('[rev ]', rev_output[0][idx], file=fw)
    #         print('[ref ]', ref_text[0][idx], file=fw)

    #     print('*' * 20, '********', '*' * 20, file=fw)

    #     for idx in range(len(rev_output[1])):
    #         print('*' * 20, 'pos sample', '*' * 20, file=fw)
    #         print('[gold]', gold_text[1][idx], file=fw)
    #         print('[raw ]', raw_output[1][idx], file=fw)
    #         print('[rev ]', rev_output[1][idx], file=fw)
    #         print('[ref ]', ref_text[1][idx], file=fw)

    #     print('*' * 20, '********', '*' * 20, file=fw)
    with open(save_file_ppl_0,'w') as sfp0:
        for idx in range(len(rev_output[0])):
            print(rev_output[0][idx],file=sfp0)
    with open(save_file_ppl_1,'w') as sfp1:
        for idx in range(len(rev_output[0])):
            print(rev_output[1][idx],file=sfp1)

def calc_temperature(temperature_config, step):
        num = len(temperature_config)
        for i in range(num):
            t_a, s_a = temperature_config[i]
            if i == num - 1:
                return t_a
            t_b, s_b = temperature_config[i + 1]
            if s_a <= step < s_b:
                k = (step - s_a) / (s_b - s_a)
                temperature = (1 - k) * t_a + k * t_b
                return temperature
def main():
    config=Config()
    train_iters, dev_iters, test_iters, vocab = load_dataset(config)
    # global_step=4800
    global_step=4950
    # name = 'Oct18145815'
    name = 'Aug29231526'
    config.save_folder=config.save_path+'/'+str(name)
    print(config.save_folder)
    save_path=config.save_folder + '/ckpts/' + str(global_step) + '_F.pth'
    model_F = StyleTransformer(config, vocab).to(config.device)
    model_F.load_state_dict(torch.load(save_path))
    print('finish load F model')
    temperature=calc_temperature(config.temperature_config,global_step)
    auto_eval(config,vocab,model_F,test_iters,global_step,temperature)
if __name__ == '__main__':
    main()
