import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


class strainDataset(Dataset):
    def __init__(self,subtype,sl):

        time_ser = pd.read_excel(f"data/{subtype}_st.xlsx")
        labels = pd.read_excel(f"data/{subtype}_nLabel.xlsx")

        self.maxLen = labels["seq"].str.len().max()
        minYearLimit = labels["year"].min() - sl

        self.datalist = []
        self.label = []

        for label in labels['strain']:
            current_id = label
            labelStrain = labels[labels['strain'] == current_id]
            current_year = labelStrain['year'].values[0]
            if current_year < minYearLimit:
                continue
            current_id = label
            labelSeq = labelStrain['seq'].values[0]
            result = []
            for i in range(sl):

                preStrain = time_ser[time_ser['strain'] == current_id]
                if preStrain.empty:
                    break
                preID = preStrain['preSeq'].values
                if len(str(preID)) < 10:
                    break
                else:
                    preID = preID[0].split(",")[0]
                    preSeq = preStrain['s1'].values[0]

                    if (len(preSeq) != 566): break
                    result.append(preSeq)
                    current_id = preID

            if len(result) == sl:
                self.label.append([self.__label_encode(result[0]),self.__label_encode(labelSeq)])  # label
                self.datalist.append(result)
        print("Construct done")

        provect = pd.read_csv("data/protVec_100d_3grams.csv", delimiter='\t')
        trigrams = list(provect['words'])
        self.__trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}
        self.__trigram_vecs = provect.loc[:, provect.columns != 'words'].values
        #
        for i in range(len(self.datalist)):
            for j in range(sl):
                self.datalist[i][j] = self.__seq_encode(self.datalist[i][j])#embedding
        self.datalist = np.array(self.datalist)
        self.label = np.array(self.label)
        # self.__scaler = MinMaxScaler(feature_range=(0, 1))
        # label_shape = self.label.shape
        # scaler = self.__scaler.fit_transform(self.label.reshape(-1,1))
        # self.label = scaler.reshape(label_shape)
        self.datalist =  self.datalist.reshape(self.datalist.shape[0],self.datalist.shape[1],-1)
        self.datalen = len(self.label)
        print("Coding completed")
    def inverse_transform(self,data):
        return self.__scaler.inverse_transform(data)
    # def __transweq(self,seq):
    #     if not isinstance(seq, str): return seq
    #     slen  = len(seq)
    #     if self.maxLen>=slen:#截取和补充
    #         seq = seq + "-" * (self.maxLen-slen)
    #     else:
    #         seq = seq[:self.maxLen]
    #     return seq

    def __label_encode(self,label):
        amino_acids = ['A', 'F', 'Q', 'R', 'T', 'Y', 'V', 'I', 'H', 'K', 'P', 'N', 'E', 'G', 'S', 'M', 'D', 'W', 'C',
                       'L', '-', 'B', 'J', 'Z', 'X']
        token2idx = {v: i for i, v in enumerate(amino_acids)}
        l = [token2idx[s] for s in label]
        l = np.array(l)
        return l

    def __seq_encode(self,seq):
        strain_embedding = []
        for i in range(0, len(seq) - 2):
            trigram = seq[i:i + 3]
            if "-" in trigram:
                tri_embedding = self.__trigram_vecs[self.__trigram_to_idx['<unk>']]
            else:
                tri_embedding = self.__trigram_vecs[self.__trigram_to_idx[trigram]]
            strain_embedding.append(tri_embedding)
        return strain_embedding


    def __getitem__(self, index):
        return torch.tensor(self.datalist[index]), torch.tensor(self.label[index])

    def __len__(self):
        return self.datalen

if __name__ == '__main__':

    dataset = strainDataset("H3N2", 10)
    print(len(dataset))
    dataloader = DataLoader(dataset=dataset,batch_size=5,shuffle=False)
    for i,(x,y) in enumerate(dataloader):
        print('| Step:', i, '| batch x: ', x.shape, '| batch y: ', y.shape)
        break


