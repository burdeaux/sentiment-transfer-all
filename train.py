import os
import time
from tracemalloc import is_tracing
from nbformat import write
import torch
import numpy as np
from torch import nn, optim
#from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from evaluator import Evaluator
from utils import tensor2text, calc_ppl, idx2onehot, add_noise, word_drop

def get_lengths(tokens, eos_idx):
    lengths = torch.cumsum(tokens == eos_idx, 1)
    lengths = (lengths == 0).long().sum(-1)
    lengths = lengths + 1 # +1 for <eos> token
    return lengths

def batch_preprocess(batch, pad_idx, eos_idx, vocab=False,reverse=False):
    batch_pos, batch_neg = batch
    diff = batch_pos.size(1) - batch_neg.size(1)
    if diff < 0:
        pad = torch.full_like(batch_neg[:, :-diff], pad_idx)
        batch_pos = torch.cat((batch_pos, pad), 1)
    elif diff > 0:
        pad = torch.full_like(batch_pos[:, :diff], pad_idx)
        batch_neg = torch.cat((batch_neg, pad), 1)

    pos_styles = torch.ones_like(batch_pos[:, 0])
    neg_styles = torch.zeros_like(batch_neg[:, 0])

    if reverse:
        batch_pos, batch_neg = batch_neg, batch_pos
        pos_styles, neg_styles = neg_styles, pos_styles
        
    tokens = torch.cat((batch_pos, batch_neg), 0)
    lengths = get_lengths(tokens, eos_idx)
    styles = torch.cat((pos_styles, neg_styles), 0)
    if vocab:
        #for collecting error sentence
        if tokens.size(1) > 32:
            print(tokens.size())
            print(lengths)
            for i in tokens:
                if i[33] != 1:
                    i.cpu()
                    sen=[vocab.itos[a] for a in i]
                    print(i)
                    print(sen)

    return tokens, lengths, styles
        
#d_step for discriminator
def d_step(config, vocab, model_F, model_D,model_N, optimizer_D, batch, temperature):
    model_F.eval()
    model_N.eval()
    pad_idx = vocab.stoi['<pad>']
    eos_idx = vocab.stoi['<eos>']
    vocab_size = len(vocab)
    loss_fn = nn.NLLLoss(reduction='none')

    inp_tokens, inp_lengths, raw_styles = batch_preprocess(batch, pad_idx, eos_idx)
    rev_styles = 1 - raw_styles
    batch_size = inp_tokens.size(0)

    raw_gen_log_probs = model_F(
        inp_tokens, 
        None,
        inp_lengths,
        raw_styles,
        generate=True,
        differentiable_decode=True,
        temperature=temperature,
    )
    rev_gen_log_probs = model_F(
        inp_tokens,
        None,
        inp_lengths,
        rev_styles,
        generate=True,
        differentiable_decode=True,
        temperature=temperature,
    )

    
    raw_gen_soft_tokens = raw_gen_log_probs.exp()
    raw_gen_lengths = get_lengths(raw_gen_soft_tokens.argmax(-1), eos_idx)

    
    rev_gen_soft_tokens = rev_gen_log_probs.exp()
    rev_gen_lengths = get_lengths(rev_gen_soft_tokens.argmax(-1), eos_idx)

        

    if config.discriminator_method == 'Multi':
        gold_log_probs = model_D(inp_tokens, inp_lengths)
        gold_labels = raw_styles + 1

        raw_gen_log_probs = model_D(raw_gen_soft_tokens, raw_gen_lengths)
        rev_gen_log_probs = model_D(rev_gen_soft_tokens, rev_gen_lengths)
        gen_log_probs = torch.cat((raw_gen_log_probs, rev_gen_log_probs), 0)
        raw_gen_labels = raw_styles + 1
        rev_gen_labels = torch.zeros_like(rev_styles)
        gen_labels = torch.cat((raw_gen_labels, rev_gen_labels), 0)
    else:
        raw_gold_log_probs = model_D(inp_tokens, inp_lengths, raw_styles)
        rev_gold_log_probs = model_D(inp_tokens, inp_lengths, rev_styles)
        gold_log_probs = torch.cat((raw_gold_log_probs, rev_gold_log_probs), 0)
        raw_gold_labels = torch.ones_like(raw_styles)
        rev_gold_labels = torch.zeros_like(rev_styles)
        gold_labels = torch.cat((raw_gold_labels, rev_gold_labels), 0)

        
        raw_gen_log_probs = model_D(raw_gen_soft_tokens, raw_gen_lengths, raw_styles)
        rev_gen_log_probs = model_D(rev_gen_soft_tokens, rev_gen_lengths, rev_styles)
        gen_log_probs = torch.cat((raw_gen_log_probs, rev_gen_log_probs), 0)
        raw_gen_labels = torch.ones_like(raw_styles)
        rev_gen_labels = torch.zeros_like(rev_styles)
        gen_labels = torch.cat((raw_gen_labels, rev_gen_labels), 0)

    
    adv_log_probs = torch.cat((gold_log_probs, gen_log_probs), 0)
    adv_labels = torch.cat((gold_labels, gen_labels), 0)
    adv_loss = loss_fn(adv_log_probs, adv_labels)
    assert len(adv_loss.size()) == 1
    adv_loss = adv_loss.sum() / batch_size
    loss = adv_loss
    
    optimizer_D.zero_grad()
    loss.backward()
    clip_grad_norm_(model_D.parameters(), 5)
    optimizer_D.step()

    model_F.train()
    model_N.train()

    return adv_loss.item()
def n_step(config, vocab, model_F, model_D,model_N, optimizer_N, batch, temperature):
    model_F.eval()
    model_D.eval()
    pad_idx = vocab.stoi['<pad>']
    eos_idx = vocab.stoi['<eos>']
    vocab_size = len(vocab)
    loss_fn = nn.NLLLoss(reduction='none')

    inp_tokens, inp_lengths, raw_styles = batch_preprocess(batch, pad_idx, eos_idx)
    rev_styles = 1 - raw_styles
    batch_size = inp_tokens.size(0)

    with torch.no_grad():
        raw_gen_log_probs = model_F(
            inp_tokens, 
            None,
            inp_lengths,
            raw_styles,
            generate=True,
            differentiable_decode=True,
            temperature=temperature,
        )
        rev_gen_log_probs = model_F(
            inp_tokens,
            None,
            inp_lengths,
            rev_styles,
            generate=True,
            differentiable_decode=True,
            temperature=temperature,
        )

    
    raw_gen_soft_tokens = raw_gen_log_probs.exp()
    raw_gen_lengths = get_lengths(raw_gen_soft_tokens.argmax(-1), eos_idx)

    
    rev_gen_soft_tokens = rev_gen_log_probs.exp()
    rev_gen_lengths = get_lengths(rev_gen_soft_tokens.argmax(-1), eos_idx)

        

    if config.discriminator_method == 'Multi':
        gold_log_probs = model_N(inp_tokens, inp_lengths)
        gold_labels = torch.zeros_like(raw_styles)#原句为0
        raw_gen_log_probs = model_N(raw_gen_soft_tokens, raw_gen_lengths)
        rev_gen_log_probs = model_N(rev_gen_soft_tokens, rev_gen_lengths)
        gen_log_probs = torch.cat((raw_gen_log_probs, rev_gen_log_probs), 0)
        raw_gen_labels = torch.ones_like(rev_styles)
        rev_gen_labels = torch.ones_like(rev_styles)
        gen_labels = torch.cat((raw_gen_labels, rev_gen_labels), 0)
    else:
        raw_gold_log_probs = model_N(inp_tokens, inp_lengths, raw_styles)
        rev_gold_log_probs = model_N(inp_tokens, inp_lengths, rev_styles)
        gold_log_probs = torch.cat((raw_gold_log_probs, rev_gold_log_probs), 0)
        raw_gold_labels = torch.zeros_like(raw_styles)#原句均为0
        rev_gold_labels = torch.zeros_like(rev_styles)
        gold_labels = torch.cat((raw_gold_labels, rev_gold_labels), 0)

        
        raw_gen_log_probs = model_N(raw_gen_soft_tokens, raw_gen_lengths)
        rev_gen_log_probs = model_N(rev_gen_soft_tokens, rev_gen_lengths)
        gen_log_probs = torch.cat((raw_gen_log_probs, rev_gen_log_probs), 0)
        raw_gen_labels = torch.ones_like(raw_styles)#生成句均为1
        rev_gen_labels = torch.ones_like(rev_styles)
        gen_labels = torch.cat((raw_gen_labels, rev_gen_labels), 0)

    
    adv_log_probs = torch.cat((gold_log_probs, gen_log_probs), 0)
    adv_labels = torch.cat((gold_labels, gen_labels), 0)
    adv_loss = loss_fn(adv_log_probs, adv_labels)
    assert len(adv_loss.size()) == 1
    adv_loss = adv_loss.sum() / batch_size
    loss = adv_loss
    
    optimizer_N.zero_grad()
    loss.backward()
    clip_grad_norm_(model_N.parameters(), 5)
    optimizer_N.step()

    model_F.train()
    model_D.train()

    return adv_loss.item()

def f_step(config, vocab, model_F, model_D, model_N,optimizer_F, batch, temperature, drop_decay,
           cyc_rec_enable=True):
    model_D.eval()
    model_N.eval()
    
    pad_idx = vocab.stoi['<pad>']
    eos_idx = vocab.stoi['<eos>']
    unk_idx = vocab.stoi['<unk>']
    vocab_size = len(vocab)
    loss_fn = nn.NLLLoss(reduction='none')

    inp_tokens, inp_lengths, raw_styles = batch_preprocess(batch, pad_idx,eos_idx,vocab)
    rev_styles = 1 - raw_styles
    batch_size = inp_tokens.size(0)
    token_mask = (inp_tokens != pad_idx).float()

    optimizer_F.zero_grad()

    # self reconstruction loss

    noise_inp_tokens = word_drop(
        inp_tokens,
        inp_lengths, 
        config.inp_drop_prob * drop_decay,
        vocab
    )
    noise_inp_lengths = get_lengths(noise_inp_tokens, eos_idx)

    slf_log_probs = model_F(
        noise_inp_tokens, 
        inp_tokens, 
        noise_inp_lengths,
        raw_styles,
        generate=False,
        differentiable_decode=False,
        temperature=temperature,
    )

    slf_rec_loss = loss_fn(slf_log_probs.transpose(1, 2), inp_tokens) * token_mask
    slf_rec_loss = slf_rec_loss.sum() / batch_size
    slf_rec_loss *= config.slf_factor
    
    slf_rec_loss.backward()

    # cycle consistency loss

    if not cyc_rec_enable:
        optimizer_F.step()
        model_D.train()
        return slf_rec_loss.item(), 0, 0
    
    gen_log_probs = model_F(
        inp_tokens,
        None,
        inp_lengths,
        rev_styles,
        generate=True,
        differentiable_decode=True,
        temperature=temperature,
    )

    gen_soft_tokens = gen_log_probs.exp()
    gen_lengths = get_lengths(gen_soft_tokens.argmax(-1), eos_idx)
    gen_stop_tokens = gen_soft_tokens.argmax(-1).detach()

    cyc_log_probs = model_F(
        gen_stop_tokens,
        inp_tokens,
        gen_lengths,
        raw_styles,
        generate=False,
        differentiable_decode=False,
        temperature=temperature,
    )

    cyc_rec_loss = loss_fn(cyc_log_probs.transpose(1, 2), inp_tokens) * token_mask
    cyc_rec_loss = cyc_rec_loss.sum() / batch_size
    cyc_rec_loss *= config.cyc_factor

    # style consistency loss

    adv_log_porbs = model_D(gen_soft_tokens, gen_lengths, rev_styles)
    if config.discriminator_method == 'Multi':
        adv_labels = rev_styles + 1
    else:
        adv_labels = torch.ones_like(rev_styles)
    adv_loss = loss_fn(adv_log_porbs, adv_labels)
    adv_loss = adv_loss.sum() / batch_size
    adv_loss *= config.adv_factor
    # n_adv_log_probs = model_N(gen_soft_tokens,gen_lengths,rev_styles)
    # n_adv_labels = torch.zeros_like(rev_styles)
    # n_adv_loss = loss_fn(n_adv_log_probs, n_adv_labels)
    # n_adv_loss = adv_loss.sum() / batch_size
    # n_adv_loss *= config.n_adv_factor
        
    (cyc_rec_loss + adv_loss).backward()
        
    # update parameters
    
    clip_grad_norm_(model_F.parameters(), 5)
    optimizer_F.step()

    model_D.train()
    model_N.train()

    return slf_rec_loss.item(), cyc_rec_loss.item(), adv_loss.item()
def write_vocab(vocab,path):
    with open(path, 'w') as f:     
        for token, index in vocab.stoi.items():
            f.write(f'{token}\n')


    return vocab
def train(config, vocab, model_F, model_D,model_N, train_iters, dev_iters, test_iters):
    optimizer_F = optim.Adam(model_F.parameters(), lr=config.lr_F, weight_decay=config.L2)
    optimizer_D = optim.Adam(model_D.parameters(), lr=config.lr_D, weight_decay=config.L2)
    optimizer_N = optim.Adam(model_N.parameters(), lr=config.lr_N, weight_decay=config.L2)

    his_d_adv_loss = []
    his_f_slf_loss = []
    his_f_cyc_loss = []
    his_f_adv_loss = []
    his_n_adv_loss = []
    
    #writer = SummaryWriter(config.log_dir)
    global_step = 0
    model_F.train()
    model_D.train()

    config.save_folder = config.save_path + '/' + str(config.exp_num)+'_'+str(config.sample_ratio)+'_'+str(time.strftime('%b%d%H%M%S', time.localtime()))
    os.makedirs(config.save_folder)
    os.makedirs(config.save_folder + '/ckpts')
    print('Save Path:', config.save_folder)
    #save vocab
    print('Save vocab first')
    vocab_path=os.path.join(config.save_folder,'vocab_obj.pth')
    torch.save(vocab,vocab_path)
    #write vocab
    print("write vocab")
    vocab_write_path=os.path.join(config.save_folder,'vocab.txt')

    write_vocab(vocab,vocab_write_path)
    print('finish writing vocab in {}'.format(vocab_write_path))

    print('Model F pretraining......')
    for i, batch in enumerate(train_iters):
        #deal with error batch (maybe come from buckteriteration,merge several sentences into one)
        #reason is some text has start"" in the sentence but no end""" 
        pad_idx = vocab.stoi['<pad>']
        eos_idx = vocab.stoi['<eos>']
        unk_idx = vocab.stoi['<unk>']

        inp_tokens, inp_lengths, raw_styles = batch_preprocess(batch, pad_idx,eos_idx,vocab)
        if inp_tokens.size(1) >=config.max_length:
            print('skip this batch')
            continue
        #pre-train stage
        if i >= config.F_pretrain_iter:
            break
        slf_loss, cyc_loss, _ = f_step(config, vocab, model_F, model_D,model_N, optimizer_F, batch, 1.0, 1.0, False)
        his_f_slf_loss.append(slf_loss)
        his_f_cyc_loss.append(cyc_loss)

        if (i + 1) % 10 == 0:
            avrg_f_slf_loss = np.mean(his_f_slf_loss)
            avrg_f_cyc_loss = np.mean(his_f_cyc_loss)
            his_f_slf_loss = []
            his_f_cyc_loss = []
            print('[iter: {}] slf_loss:{:.4f}, rec_loss:{:.4f}'.format(i + 1, avrg_f_slf_loss, avrg_f_cyc_loss))

    print('Training start......')

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
    batch_iters = iter(train_iters)
    while True:
        drop_decay = calc_temperature(config.drop_rate_config, global_step)
        temperature = calc_temperature(config.temperature_config, global_step)
        batch = next(batch_iters)
        
        for _ in range(config.iter_D):
            batch = next(batch_iters)
           #deal with error batch (maybe come from buckteriteration,merge several sentences into one)
            pad_idx = vocab.stoi['<pad>']
            eos_idx = vocab.stoi['<eos>']
            unk_idx = vocab.stoi['<unk>']

            inp_tokens, inp_lengths, raw_styles = batch_preprocess(batch, pad_idx,eos_idx,vocab)
            if inp_tokens.size(1) >=config.max_length:
                print('skip this batch')
                continue


            d_adv_loss = d_step(
                config, vocab, model_F, model_D, model_N,optimizer_D, batch, temperature
            )
            his_d_adv_loss.append(d_adv_loss)
            
        for _ in range(config.iter_F):
            batch = next(batch_iters)
             #deal with error batch (maybe come from buckteriteration,merge several sentences into one)
            pad_idx = vocab.stoi['<pad>']
            eos_idx = vocab.stoi['<eos>']
            unk_idx = vocab.stoi['<unk>']

            inp_tokens, inp_lengths, raw_styles = batch_preprocess(batch, pad_idx,eos_idx,vocab)
            if inp_tokens.size(1) >=config.max_length:
                print('skip this batch')
                continue


            f_slf_loss, f_cyc_loss, f_adv_loss = f_step(
                config, vocab, model_F, model_D, model_N,optimizer_F, batch, temperature, drop_decay
            )
            his_f_slf_loss.append(f_slf_loss)
            his_f_cyc_loss.append(f_cyc_loss)
            his_f_adv_loss.append(f_adv_loss)
        # for _ in range(config.iter_N):
        #     batch = next(batch_iters)
        #     n_adv_loss = n_step(
        #         config,vocab,model_F,model_D,model_N,optimizer_N,batch,temperature
        #     )
        #     his_n_adv_loss.append(n_adv_loss)
        
        global_step += 1
        #writer.add_scalar('rec_loss', rec_loss.item(), global_step)
        #writer.add_scalar('loss', loss.item(), global_step)
            
            
        if global_step % config.log_steps == 0:
            avrg_d_adv_loss = np.mean(his_d_adv_loss)
            avrg_f_slf_loss = np.mean(his_f_slf_loss)
            avrg_f_cyc_loss = np.mean(his_f_cyc_loss)
            avrg_f_adv_loss = np.mean(his_f_adv_loss)
            avrg_n_adv_loss = np.mean(his_n_adv_loss)
            log_str = '[iter {}] d_adv_loss: {:.4f}  ' + \
                      'f_slf_loss: {:.4f}  f_cyc_loss: {:.4f}  ' + \
                      'f_adv_loss: {:.4f}  n_adv_loss: {:.4f}   temp: {:.4f}  drop: {:.4f}'
            print(log_str.format(
                global_step, avrg_d_adv_loss,
                avrg_f_slf_loss, avrg_f_cyc_loss, avrg_f_adv_loss,avrg_n_adv_loss,
                temperature, config.inp_drop_prob * drop_decay
            ))
                
        if global_step % config.eval_steps == 0:
            his_d_adv_loss = []
            his_f_slf_loss = []
            his_f_cyc_loss = []
            his_f_adv_loss = []
            his_n_adv_loss = []

            #save model
            torch.save(model_F.state_dict(), config.save_folder + '/ckpts/' + str(global_step) + '_F.pth')
            torch.save(model_D.state_dict(), config.save_folder + '/ckpts/' + str(global_step) + '_D.pth')
            #here only use dev_iters for comparing with other models
            auto_eval(config, vocab, model_F, dev_iters, global_step, temperature)
            #for path, sub_writer in writer.all_writers.items():
            #    sub_writer.flush()


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
            #check inp_tokens
            # if inp_tokens.size(1) > 1:
            #     print(inp_tokens.size())
            #     for i in inp_tokens:
            #         if i[1] != 1:
            #             i.cpu()
            #             sen=[vocab.itos[a] for a in i]
            #             print(i)
            #             print(sen)
            #deal with error batch problem
            if inp_tokens.size(1)>32:
                print('skip wrong batch in test')
                continue
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
    try:
        bleu_neg = evaluator.yelp_ref_bleu_0(rev_output[0])
        bleu_pos = evaluator.yelp_ref_bleu_1(rev_output[1])
    except:
        #for other test data
        print('this is not reference test,only test bleu_self')
        bleu_neg=0
        bleu_pos=0
        empty_text_0=[0 for i in range(len(rev_output[0]))]
        empty_text_1=[0 for i in range(len(rev_output[1]))]
        
        ref_text = [empty_text_0,empty_text_1]
        # print(ref_text)

    bleu_self = evaluator.self_bleu_b(gold_text[0]+gold_text[1],rev_output[0]+rev_output[1],)
    # ppl_neg = evaluator.yelp_ppl(rev_output[0])
    # ppl_pos = evaluator.yelp_ppl(rev_output[1])
    
    #use additional evaluator for evaluation
    #need translate the sentence in test.txt and show its original attribute in test.attr

    
    

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
              'bleu_self:{:.4f}'+\
          'ppl_pos: {:.4f} ppl_neg: {:.4f}\n').format(
              acc_pos, acc_neg, bleu_pos, bleu_neg, bleu_self,0, 0,
    ))

    
    # save output
    save_file = config.save_folder + '/' + str(global_step) + '.txt'
    eval_log_file = config.save_folder + '/eval_log.txt'
    eval_output_file=config.save_folder +'/' +str(global_step)+'_test.txt'
    eval_output_attr=config.save_folder +'/'+ str(global_step)+'.attr'
    
    with open(eval_log_file, 'a') as fl:
        print(('iter{:5d}:  acc_pos: {:.4f} acc_neg: {:.4f} ' + \
               'bleu_pos: {:.4f} bleu_neg: {:.4f} ' + \
                'bleu_self: {:.4f}' + \
               'ppl_pos: {:.4f} ppl_neg: {:.4f}\n').format(
            global_step, acc_pos, acc_neg, bleu_pos, bleu_neg, bleu_self,0, 0,
        ), file=fl)
    with open(save_file, 'w') as fw:
        print(('[auto_eval] acc_pos: {:.4f} acc_neg: {:.4f} ' + \
               'bleu_pos: {:.4f} bleu_neg: {:.4f} ' + \
                'bleu_self:{:.4f}'+\
               'ppl_pos: {:.4f} ppl_neg: {:.4f}\n').format(
            acc_pos, acc_neg, bleu_pos, bleu_neg, bleu_self,0, 0,
        ), file=fw)

        for idx in range(len(rev_output[0])):
            print('*' * 20, 'neg sample', '*' * 20, file=fw)
            print('[gold]', gold_text[0][idx], file=fw)
            print('[raw ]', raw_output[0][idx], file=fw)
            print('[rev ]', rev_output[0][idx], file=fw)
            print('[ref ]', ref_text[0][idx], file=fw)

        print('*' * 20, '********', '*' * 20, file=fw)

        for idx in range(len(rev_output[1])):
            print('*' * 20, 'pos sample', '*' * 20, file=fw)
            print('[gold]', gold_text[1][idx], file=fw)
            print('[raw ]', raw_output[1][idx], file=fw)
            print('[rev ]', rev_output[1][idx], file=fw)
            print('[ref ]', ref_text[1][idx], file=fw)

        print('*' * 20, '********', '*' * 20, file=fw)
    
    fo=open(eval_output_file, 'w') 
    fa=open(eval_output_attr,'w')

    for idx in range(len(rev_output[0])):
            fo.write(rev_output[0][idx].strip()+'\n')
            fa.write('negative\n')
    for idx in range(len(rev_output[1])):
            fo.write(rev_output[1][idx].strip()+'\n')
            fa.write('positive\n')

    fo.close()
    fa.close()    
    model_F.train()
