# Data preprocessing such that all rows in a given utterance is present in the data. Thus we have contextual information that can be used by LSTMs
# This treats each utterance as independent entity, relationship between dialogues and utterances not preserved
# Relationship between utterances of the same dialogues may be required once I start with multi-party full on
import numpy as np
import pandas as pd
import csv
import time
import VisualDataloader
import pickle


action_units = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14',
                    'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU45']

class_to_idx = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}


def process_action_units(file, utterance):
    per_utt = []
    for row in range(len(utterance)):
        per_row = []
        for au in action_units:
            per_row.extend([utterance[au + '_c'][row] * utterance[au + '_r'][row]])
        per_utt.append(np.asarray(per_row))

    return np.asarray(per_utt)

def process_utterance_product(file):
    filename = file.split('/')[-1].split('.')[0]
    try:
        utterance = pd.read_csv(file, sep=r',', skipinitialspace=True)
        processed_au = process_action_units(file, utterance)
        return processed_au
    except IOError:
        print("{0} not found".format(filename))


def process_utterance_concat(file):
    utterance = pd.read_csv(file, sep=r',', skipinitialspace=True)
    per_utt = []
    for row in range(len(utterance)):
        per_row = []
        for au in action_units:
            per_row.extend(np.concatenate(utterance[au + '_c'][row], utterance[au + '_r'][row]))
        per_utt.append(np.asarray(per_row))

    return np.asarray(per_utt)


def au_extraction(filename):
    # file = filename.split('/')[-1].split('.')[0]
    # print('Processing file:', file)
    utt_feat = process_utterance_product(filename)

    try:
        if len(utt_feat) > 0:
            processed_feat = utt_feat   #Extract features for an utterance here or just pass the features extracted from an utterance
        else:
            processed_feat = np.zeros((50, 17))
    except TypeError:
        processed_feat = np.zeros((50, 17))

    return processed_feat


class DialogIter():
    def __init__(self, data, labels, batch_size, MODE, split, shuffle:bool=False):
        x = pickle.load(open("./data/pickles/data_{}.p".format(MODE.lower()), "rb"))
        revs, W, word_idx_map, vocab, _, label_index = x[0], x[1], x[2], x[3], x[4], x[5]
        print("Labels used for this classification: ", label_index)

        data_visual = {}
        for i in range(len(revs)):
            utterance_id = revs[i]['dialog'] + "_" + revs[i]['utterance']
            # label = label_index[revs[i]['y']]

            file = 'dia' + str(revs[i]['dialog']) + '_utt' + str(revs[i]['utterance'])

            if revs[i]['split'] == split:
                data_visual[utterance_id] = au_extraction(
                    '../../../ConfidenceFilteringTrain/' + file + '/' + file + '_conf' + '.csv')

        batch_num = int(np.ceil(len(data) / batch_size))
        index_array = list(range(len(data)))
        if shuffle:
            np.random.shuffle(index_array)

        for i in range(batch_num):
            indices = index_array[i * batch_size:(i + 1) * batch_size]
            batch_data = [data[idx] for idx in indices]
            batch_labels = [class_to_idx[labels[idx]] for idx in indices]
            yield (batch_data, batch_labels)


def dialog_iter(split, batch_size, MODE, shuffle:bool=False):
    x = pickle.load(open("./data/pickles/data_emotion.p", "rb"))
    revs, W, word_idx_map, vocab, _, label_index = x[0], x[1], x[2], x[3], x[4], x[5]
    # print("Labels used for this classification: ", label_index)

    data_visual = {}

    for i in range(len(revs)):
        utterance_id = revs[i]['dialog'] + "_" + revs[i]['utterance']
        # label = label_index[revs[i]['y']]
        file = 'dia' + str(revs[i]['dialog']) + '_utt' + str(revs[i]['utterance'])
        if revs[i]['split'] == split:
            data_visual[utterance_id] = au_extraction('../../../ConfidenceFiltering'+split.upper()+'/' + file + '/' + file + '_conf' + '.csv')

    batch_num = int(np.ceil(len(data_visual) / batch_size))
    # index_array = list(range(len(data_visual)))
    dia_keys = list(data_visual.keys())
    if shuffle:
        np.random.shuffle(dia_keys)

    for i in range(batch_num):
        batch_data = []
        indices = dia_keys[i*batch_size:(i+1)*batch_size]
        for index in indices:
            batch_data.append(data_visual[index])
        yield (batch_data, indices)