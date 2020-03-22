"""
Script to use the pretrained model to generate embeddings for visual data.
The extracted embeddings are stored in a pickle file.
"""

import numpy as np
import pandas as pd
import csv
import time
import sys
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
import logging
import data_helpers_visual as vis
from collections import Counter, defaultdict
import pickle

embedding_size = 17
batch_size = 32
max_utts = 33

def to_tensor(sequences, embed_size=17, pad_index=0):
    max_sequence_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    sequence_array = np.zeros((batch_size, max_sequence_len, embed_size), dtype=np.float64)
    sequence_array.fill(pad_index)

    for e_id in range(batch_size):
        seq_i = sequences[e_id]
        sequence_array[e_id, :len(seq_i)] = seq_i
    sequence_array = torch.from_numpy(sequence_array).cuda()

    return sequence_array


class RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, out_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.rnn1 = nn.RNN(input_size=embed_size, hidden_size=hidden_size, num_layers=4, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, out_size)

    def forward(self, x):
        utt_batch = to_tensor(x, self.embedding_size, pad_index=-1)
        packed_embeddings = torch.nn.utils.rnn.pack_padded_sequence(utt_batch,
                                                                    lengths=torch.tensor([len(seq) for seq in x]),
                                                                    batch_first=True, enforce_sorted=False)
        out, hidden = self.rnn1(packed_embeddings.float())
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # out = self.act1(self.fc1(hidden[-1]))
        # out = self.fc2(out)
        out = self.act1(self.fc1(hidden[-1]))
        return out

def get_dialogue_text_embs(data, dialogue_ids):
    key = list(data.keys())[0]
    # pad = [0] * len(data[key])
    pad = np.zeros((len(data[key])))

    def get_emb(dialogue_id, local_data):
        dialogue_text = {}
        for vid in dialogue_id.keys():
            local_text = []
            for utt in dialogue_id[vid]:
                sample = local_data[vid + "_" + str(utt)][:]
                sample = sample.detach().cpu().numpy()
                local_text.append(np.asarray(sample))
            for _ in range(max_utts - len(local_text)):
                local_text.append(pad[:])
            dialogue_text[vid] = np.asarray(local_text[:max_utts])
        return dialogue_text

    dialogue_features = get_emb(dialogue_ids, data)
    return dialogue_features


def get_dialogue_ids(keys):
    ids = defaultdict(list)
    for key in keys:
        ids[key.split("_")[0]].append(int(key.split("_")[1]))
    for ID, utts in ids.items():
        ids[ID] = [str(utt) for utt in sorted(utts)]
    return ids


def extract_visual_embeddings(model, split, mode):
    batch_iter = vis.dialog_iter(split, batch_size, MODE=mode, shuffle=False)

    # weights = [1.00, 3.90, 17.82, 6.69, 2.72, 17.21, 4.25]
    # class_weights = torch.FloatTensor(weights).cuda()
    # criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    generated_embeddings = {}

    with torch.no_grad():
        for batch_id, batch in enumerate(batch_iter):
            x_data, indices = batch[0], batch[1]
            embed = model(x_data)
            for emb, index in zip(embed, indices):
                generated_embeddings[index] = emb

    dialogue_ids = get_dialogue_ids(generated_embeddings.keys())
    final_embeddings = get_dialogue_text_embs(generated_embeddings, dialogue_ids)

    return final_embeddings



def extract_emb_wrapper(mode):
    OUTPUT_PATH = "./data/pickles/visual_emotion.pkl"
    model = RNN(embedding_size, hidden_size=256, out_size=7)
    model = model.cuda()

    model.load_state_dict(torch.load('./data/checkpoints/model79.pt'))
    model = model.eval()

    logging.debug('MODEL')
    logging.debug(model)
    train_embeddings = extract_visual_embeddings(model, split="train", mode=mode)
    print('Extracted train embeddings. Computing validation embeddings next...')
    valid_embeddings = extract_visual_embeddings(model, split="val", mode=mode)
    print('Extracted validation embeddings. Computing test embeddings next...')
    test_embeddings = extract_visual_embeddings(model, split="test", mode=mode)

    print('Storing the pickle file')
    pickle.dump([train_embeddings, valid_embeddings, test_embeddings], open(OUTPUT_PATH, "wb"))

    print('TRAIN: Size of dictionary is {0}, shape of each item in the dictionary {1}'.format(len(train_embeddings), train_embeddings['0'].shape))
    print('VAL: Size of dictionary is {0}, shape of each item in the dictionary {1}'.format(len(valid_embeddings), valid_embeddings['0'].shape))
    print('TEST: Size of dictionary is {0}, shape of each item in the dictionary {1}'.format(len(test_embeddings), test_embeddings['0'].shape))

    print('Completed phase 4 out of 8')
    """
    Phase 1 - Plan for visual features. Developing the model for visual features
    Phase 2 - Data pipeline to read from data_emotion.p file and get dialog ids to convert the dictionary of utterances to another dictionary of 
    all utterances in a dialog.
    Phase 3 - Run the features through the pre-trained model to extract embeddings. Pipeline to convert all the embeddings to numpy array and pad 
    to have  equal lengths.
    Phase 4 - Generating the embeddings and writing them to pickle file for train, val and test data. 
    Phase 5 - Reading the pickle files (similar to how its done in audio modality) and get all the data ready for training the multimodal network.
    Phase 6 - Completing the data pipeline required for training the multimodal network.
    Phase 7 - Training the multimodal network
    Phase 8 - Results analysis
    """


    # Binary and multiclass
    # More experiments with baseline
