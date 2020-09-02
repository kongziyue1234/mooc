import torch as t
from utils import spectrogram2wav
from scipy.io.wavfile import write
import hyperparams as hp
from text import text_to_sequence
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def split_title_line(title_text, max_words=5):
    """
    A function that splits any string based on specific character
    (returning it with the string), with maximum number of words on it
    """
    seq = title_text.split()
    return '\n'.join([' '.join(seq[i:i + max_words]) for i in range(0, len(seq), max_words)])

def plot_alignment(alignment, path, title=None, split_title=False, max_len=None):
    if max_len is not None:
        alignment = alignment[:, :max_len]
    #print(alignment.shape())
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'

    if split_title:
        title = split_title_line(title)

    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()


def load_checkpoint(step, model_name="transformer"):
    state_dict = t.load('D:\陈曦\数据\checkpoint\checkpoint_%s_%d.pth.tar' % (model_name, step))
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict


def synthesis(text, args, num):
    m = Model()
    m_post = ModelPostNet()

    m.load_state_dict(load_checkpoint(args.restore_step1, "transformer"))
    m_post.load_state_dict(load_checkpoint(args.restore_step2, "postnet"))

    text = np.asarray(text_to_sequence(text, [hp.cleaners]))
    text = t.LongTensor(text).unsqueeze(0)
    text = text.cuda()
    mel_input = t.zeros([1, 1, 80]).cuda()
    pos_text = t.arange(1, text.size(1) + 1).unsqueeze(0)
    pos_text = pos_text.cuda()

    m = m.cuda()
    m_post = m_post.cuda()
    m.train(False)
    m_post.train(False)


    pbar = tqdm(range(args.max_len))
    with t.no_grad():
        for i in pbar:
            pos_mel = t.arange(1, mel_input.size(1) + 1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = m.forward(text, mel_input, pos_text, pos_mel)
            # print('mel_pred==================',mel_pred.shape)
            # print('postnet_pred==================', postnet_pred.shape)
            mel_input = t.cat([mel_input, postnet_pred[:, -1:, :]], dim=1)
            #print(postnet_pred[:, -1:, :])
            #print(t.argmax(attn[1][1][i]).item())
            #print('mel_input==================', mel_input.shape)

    # #直接用真实mel测试postnet效果
    #aa = t.from_numpy(np.load('D:\SSHdownload\\000101.pt.npy')).cuda().unsqueeze(0)
    # # print(aa.shape)
    mag_pred = m_post.forward(postnet_pred)
    #real_mag = t.from_numpy((np.load('D:\SSHdownload\\003009.mag.npy'))).cuda().unsqueeze(dim=0)
    #wav = spectrogram2wav(postnet_pred)

    #print('shappe============',attn[2][0].shape)
    # count = 0
    # for j in range(4):
    #     count += 1
    #     attn1 = attn[0][j].cpu()
    #     plot_alignment(attn1, path='./training_loss/'+ str(args.restore_step1)+'_'+str(count)+'_'+'S'+str(num)+'.png', title='sentence'+str(num))

    attn1=attn[0][1].cpu()
    plot_alignment(attn1, path='./training_loss/'+ str(args.restore_step1)+'_'+'S'+str(num)+'.png', title='sentence'+str(num))

    wav = spectrogram2wav(mag_pred.squeeze(0).cpu().detach().numpy())
    write(hp.sample_path + '/' + str(args.restore_step1) + '-' + "test" + str(num) + ".wav", hp.sr, wav)
    # num += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step1', type=int, help='Global step to restore checkpoint', default=300000)
    parser.add_argument('--restore_step2', type=int, help='Global step to restore checkpoint', default=300000)
    parser.add_argument('--max_len', type=int, help='Global step to restore checkpoint', default=300)

    args = parser.parse_args()
    # global num
    # num = 0
    #synthesis("nan4 wu2 sa1 dan4 ta1,su1 jia1 duo1 ye1", args, 10)
    #synthesis("ni2 xi3 wan2 shou3 zen3 me5 bu4 ca1 gan1 dou1 shi4 shui3",args, 1)
    #synthesis("ming4 tian hui3 xia2 yu3 ma", args, 9)
    #synthesis("er4 shi2 guo2 ji2 tuan2 ling3 dao3 ren2.", args, 2)
    #synthesis("ye3 dui4 wen3 ding4 zhong1 guo2 de wai4 mao4 qi3 dao4 le zhong4 yao4 zhi1 cheng1.",args, 3)
    #
    # # # #
    #synthesis("ci3 hou4 tou1 dao4 qing2 kuang4 you2 suo3 hao2 zhuan3 dan4 reng2 shi2 you3 cun1 min2 fan1 yue4 zha4 lan5.", args, 4)
    # # #you should 有重复现象
    #synthesis("li3 ke4 qiang2 zhi3 chu1,zhong1 fang1 yuan4 jiang1 yi1 dai4 yi1 lu4",args, 5)
    #synthesis("nong2 min2 yu3 fa3 niu2 di3 jiao3 yang2 ken3 cao3 hao2 zi5 han1 han1 qun2 yu2 huan1 wu3 zhong4 ya1 xi1 dang4", args, 6)
    #synthesis("wei4 xi1 yin3 you2 yong3 ai4 hao4 zhe3 ta1 men5 fa1 shou4 nian2 ka3 yue4 ka3 qing2 lv2 ka3 he2 jia1 huan1 ka3 jin1 ka3", args, 7)
    #
    #synthesis("ning3 kai1 su4 liao4 gai4 er2 yi2 kan4 ai4 you4 shi4 la4 feng1 de5 mu4 sai1 er2 tong2 yang4 ye3 dou1 xian4 zai4 ping2 kou3 nei4",args, 8)
    #synthesis(" ni3 men hao3 . cong2 jin1 tian1 kai1 shi3 	 wo3 men jiu4 yao4 zheng4 shi4 kai1 shi3 xue2 xi2 xin1 li3 zi1 xun2 le .",args, 68)
    f = open('D:/陈曦/dataset/result_pinyin.txt', 'r', encoding='utf-8')
    lines = f.readlines()
    i = 1
    for line in lines:
        synthesis(line,args,i)
        i = i + 1
