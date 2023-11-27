import os
import time
from collections import Counter
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Levenshtein import hamming
from matplotlib import pyplot as plt
from rouge import Rouge
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch_geometric.transforms as T
from tsData import strainDataset
from graphData import graphDataset
from models.model import Vocab,FluPMT,UncertaintyLoss
from torch.optim.lr_scheduler import StepLR,MultiStepLR


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


def train(train_loader,gData,testDataloader,g_testData, num_epochs,subtype,ts):

    model = FluPMT(FeatureDim=56400, ExpertOutDim=512, TaskExpertNum=1, CommonExpertNum=1
                   , out_len=pred_len, seq_len=5, label_len=label_len).to(device)
    weighted_loss_func = UncertaintyLoss(2)
    weighted_loss_func.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, list(model.parameters()) + list(weighted_loss_func.parameters())), lr=0.001,
        weight_decay=1e-4, eps=1e-09, betas=(0.9, 0.98)) 
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    loss_func1 = torch.nn.MSELoss().to(device)
    loss_func2 = torch.nn.MSELoss().to(device)

    train_loss = []
    model.train()
    t0 = time.time()
    print(f"{subtype}_{ts}===================")
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))

        for batch_idx, (encoder_inputs, decoder_targets) in loop:

            encoder_inputs, decoder_targets = encoder_inputs.float().to(device), decoder_targets.float()

            dec_inp = torch.zeros([decoder_targets.shape[0], pred_len, decoder_targets.shape[-1]]).float()
            dec_inp = torch.cat([decoder_targets[:, :label_len, :], dec_inp], dim=1).float().to(device)
            pred1, pred2 = model(encoder_inputs, dec_inp,gData)
            decoder_targets = decoder_targets[:, -pred_len:, :].to(device)
            loss1 = loss_func1(pred1, decoder_targets)

            edge = gData.edge_attr.float()
            loss2 = loss_func2(pred2,edge)
            loss = weighted_loss_func(loss1, loss2)
            # loss = loss1*torch.exp(-log_T_a)+loss2*torch.exp(-log_T_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(uloss=loss.item(),mloss =loss1.item() )
        scheduler.step()
        train_loss.append([loss1.item(), loss2.item(), loss.item()])
    t1 = time.time()
    print("train time",t1 - t0)

    train_loss = np.array(train_loss)

    plt.plot(train_loss[:, 0], label='loss1')
    # plt.plot(train_loss[:, 1], label='loss2')
    # plt.plot(train_loss[:, 2], label='UncertaintyLoss')
    plt.ylabel('train loss')
    plt.legend()
    plt.show()
    # test
    t0 = time.time()
    bleu_4_scores, rouger_score, hamm, ress = evaluate(testDataloader, g_testData, model,subtype,ts, bleu_k=4)
    torch.save(model.state_dict(), f'FLM{subtype}_{ts}.pt')
    print(f"==============={subtype}_{ts}==========seed5=========")
    print("bleu4", np.mean(bleu_4_scores))
    print("rouger_score", np.mean(rouger_score))
    print("hamm", np.mean(hamm))
    t1 = time.time()
    print("test time",t1 - t0)
    with open('FLM.txt', 'a') as file:
        result = f"==============={subtype}_{ts}=============\n"
        result = result+ f"bleu4:{np.mean(bleu_4_scores)},rouger_score:{np.mean(rouger_score)},hamm:{np.mean(hamm)}\n"
        file.write(result)


def bleu(label, pred, k=4):
    assert len(pred) >= k
    score = math.exp(min(0, 1 - len(label) / len(pred)))
    for n in range(1, k + 1):
        hashtable = Counter([' '.join(label[i:i + n]) for i in range(len(label) - n + 1)])
        num_matches = 0
        for i in range(len(pred) - n + 1):
            ngram = ' '.join(pred[i:i + n])
            if ngram in hashtable and hashtable[ngram] > 0:
                num_matches += 1
                hashtable[ngram] -= 1
        score *= math.pow(num_matches / (len(pred) - n + 1), math.pow(0.5, n))
    return score


def evaluate(test_loader,g_testData, model,subtype,ts, bleu_k):
    rouger = Rouge()
    bleu_scores = []
    rouger_score = []
    translation_results = []
    hamm = []
    model.eval()
    for src_seq, tgt_seq in test_loader:
        encoder_inputs = src_seq.float().to(device)
        dec_inp = torch.zeros([tgt_seq.shape[0], pred_len, tgt_seq.shape[-1]])
        dec_inp = torch.cat([tgt_seq[:, :label_len, :], dec_inp], dim=1).float().to(device)
        output,_ = model(encoder_inputs, dec_inp,g_testData)
        output = torch.round(output).int()

        his = tgt_seq[:, :label_len, :].reshape(-1, tgt_seq.shape[-1]).cpu().detach().numpy().squeeze().astype(int).tolist()
        his = vaca[his]

        tgt_seq = tgt_seq[:, -pred_len:, :].reshape(-1, tgt_seq.shape[-1]).cpu().detach().numpy().squeeze().astype(int).tolist()
        tgt_seq = vaca[tgt_seq]

        output = output.reshape(-1, output.shape[-1]).cpu().detach().numpy().squeeze().tolist()
        pred_seq = vaca[output]

        translation_results.append([his,tgt_seq, pred_seq])

        if len(pred_seq) >= bleu_k:
            bleu_scores.append(bleu(tgt_seq, pred_seq, k=bleu_k))
            if len(tgt_seq)>len(pred_seq):
                tgt_seq = tgt_seq[:len(pred_seq)]
            rouger_score.append(rouger.get_scores(tgt_seq, pred_seq, avg=True)["rouge-l"]["f"])
            hamm.append(hamming(tgt_seq, pred_seq))
    # np.save(f"stafre{subtype}_{ts}_.npy", np.array(translation_results))
    return bleu_scores,rouger_score,hamm, translation_results


if __name__ == '__main__':
    subtypes = ["H1N1","H3N2"]
    vaca = Vocab()
    EPOCH = 80
    pred_len = 1
    label_len = 1
    timeSteps = [5,10,15]

    for s in subtypes:
        for ts in timeSteps:
            gData = graphDataset(s)
            gData = gData.data
            tsData = strainDataset(s, ts)
            datalen = len(tsData)
            trainL = int(datalen * 0.9)
            trainData = TensorDataset(tsData[:trainL][0], tsData[:trainL][1])
            testData = TensorDataset(tsData[trainL:][0], tsData[trainL:][1])
            trainDataloader = DataLoader(dataset=trainData, batch_size=64, shuffle=True)
            testDataloader = DataLoader(dataset=testData, batch_size=1)

            transform = T.Compose([
                T.NormalizeFeatures(),
                T.ToDevice(device),
                T.RandomLinkSplit(num_val=0.05, num_test=0.15, is_undirected=True,
                                  split_labels=True, add_negative_train_samples=False),
            ])
            g_trainData, _, g_testData = transform(gData)
            train(trainDataloader, g_trainData,testDataloader,g_testData, EPOCH,s,ts)



