from sklearn import preprocessing
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
import random

class graphDataset(Dataset):
    def __init__(self,subtype):
        path = f"data/{subtype}_antigenicDistance.xlsx"
        strain = pd.read_excel(path)
        strain["strain1"] = strain["strain1"].str.lower()
        strain["strain2"] = strain["strain2"].str.lower()
        lables = preprocessing.LabelEncoder()
        lables.fit(np.unique(list(strain["strain1"].values)+list(strain["strain2"].values)))
        strain["id1"] = lables.transform(list(strain["strain1"].values))
        strain["id2"] = lables.transform(list(strain["strain2"].values))
        strain.sort_values(by=['id1', 'id2'],inplace=True)
        "DATA edge"
        oriN = strain["id1"].values
        endN = strain["id2"].values
        on = np.hstack((oriN,endN))
        en = np.hstack((endN,oriN))
        edge_index = torch.tensor(np.array([on,en]), dtype=torch.long)
        "edge_attr"
        ea = np.hstack((strain["distance"].values,strain["distance"].values))
        ea_min = np.min(ea)
        shifted_arr = ea - ea_min + 1
        ea_min = np.min(shifted_arr)
        ea_max = np.max(shifted_arr)
        ea_normalized = (shifted_arr - ea_min) / (ea_max - ea_min)
        edge_attr  = torch.tensor(ea_normalized, dtype=torch.float)
        "x"
        provect = pd.read_csv("data/protVec_100d_3grams.csv", delimiter='\t')
        # strain = pd.read_csv(path3, names=['seq', 'description'])
        # strain["description"] = strain["description"].str.lower()
        trigrams = list(provect['words'])
        trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}
        trigram_vecs = provect.loc[:, provect.columns != 'words'].values
        x = []

        for strain_name in lables.classes_:
            if len(strain[strain["strain1"]==strain_name]["seq1"])>0:
                seq = strain[strain["strain1"]==strain_name]["seq1"].values[0]
            else:
                seq = strain[strain["strain2"] == strain_name]["seq2"].values[0]
            strain_embedding = []
            for i in range(0, len(seq) - 2):
                trigram = seq[i:i + 3]
                if "-" in trigram:
                    tri_embedding = trigram_vecs[trigram_to_idx['<unk>']]
                else:
                    tri_embedding = trigram_vecs[trigram_to_idx[trigram]]
                strain_embedding.append(tri_embedding)
            x.append(strain_embedding)
        x = np.array(x)
        x = torch.tensor(x.reshape(x.shape[0],-1), dtype=torch.float)
        self.data = Data(x=x,edge_index=edge_index,edge_attr=edge_attr)
