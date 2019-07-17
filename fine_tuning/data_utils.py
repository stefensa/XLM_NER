import os
import numpy as np
import torch
from logging import getLogger
import pickle
import torch.utils.data as Data


logger = getLogger()


BOS_WORD = '<s>'
EOS_WORD = '</s>'
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'

SPECIAL_WORD = '<special%i>'
SPECIAL_WORDS = 10

SEP_WORD = SPECIAL_WORD % 0
MASK_WORD = SPECIAL_WORD % 1


class Dictionary(object):

    def __init__(self, id2word, word2id, counts):
        assert len(id2word) == len(word2id) == len(counts)
        self.id2word = id2word
        self.word2id = word2id
        self.counts = counts
        self.bos_index = word2id[BOS_WORD]
        self.eos_index = word2id[EOS_WORD]
        self.pad_index = word2id[PAD_WORD]
        self.unk_index = word2id[UNK_WORD]
        self.check_valid()

    def __len__(self):
        """
        Returns the number of words in the dictionary.
        """
        return len(self.id2word)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.id2word[i]

    def __contains__(self, w):
        """
        Returns whether a word is in the dictionary.
        """
        return w in self.word2id

    def __eq__(self, y):
        """
        Compare this dictionary with another one.
        """
        self.check_valid()
        y.check_valid()
        if len(self.id2word) != len(y):
            return False
        return all(self.id2word[i] == y[i] for i in range(len(y)))

    def check_valid(self):
        """
        Check that the dictionary is valid.
        """
        assert self.bos_index == 0
        assert self.eos_index == 1
        assert self.pad_index == 2
        assert self.unk_index == 3
        assert all(self.id2word[4 + i] == SPECIAL_WORD % i for i in range(SPECIAL_WORDS))
        assert len(self.id2word) == len(self.word2id) == len(self.counts)
        assert set(self.word2id.keys()) == set(self.counts.keys())
        for i in range(len(self.id2word)):
            assert self.word2id[self.id2word[i]] == i
        last_count = 1e18
        for i in range(4 + SPECIAL_WORDS, len(self.id2word) - 1):
            count = self.counts[self.id2word[i]]
            assert count <= last_count
            last_count = count

    def index(self, word, no_unk=False):
        """
        Returns the index of the specified word.
        """
        if no_unk:
            return self.word2id[word]                        # 若word不存在词典中，报错
        else:
            return self.word2id.get(word, self.unk_index)    # 若word不存在词典中，默认使用unk来表示

    def max_vocab(self, max_vocab):
        """
        Limit the vocabulary size.
        """
        assert max_vocab >= 1
        init_size = len(self)
        self.id2word = {k: v for k, v in self.id2word.items() if k < max_vocab}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.counts = {k: v for k, v in self.counts.items() if k in self.word2id}
        self.check_valid()
        logger.info("Maximum vocabulary size: %i. Dictionary size: %i -> %i (removed %i words)."
                    % (max_vocab, init_size, len(self), init_size - len(self)))

    def min_count(self, min_count):
        """
        Threshold on the word frequency counts.
        """
        assert min_count >= 0
        init_size = len(self)
        self.id2word = {k: v for k, v in self.id2word.items() if self.counts[self.id2word[k]] >= min_count or k < 4 + SPECIAL_WORDS}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.counts = {k: v for k, v in self.counts.items() if k in self.word2id}
        self.check_valid()
        logger.info("Minimum frequency count: %i. Dictionary size: %i -> %i (removed %i words)."
                    % (min_count, init_size, len(self), init_size - len(self)))

    @staticmethod
    def read_vocab(vocab_path):
        """
        Create a dictionary from a vocabulary file.
        """
        skipped = 0
        assert os.path.isfile(vocab_path), vocab_path
        word2id = {BOS_WORD: 0, EOS_WORD: 1, PAD_WORD: 2, UNK_WORD: 3}
        for i in range(SPECIAL_WORDS):    # 添加数字1-10的特殊字符
            word2id[SPECIAL_WORD % i] = 4 + i

        counts = {k: 0 for k in word2id.keys()}
        f = open(vocab_path, 'r', encoding='utf-8')
        for i, line in enumerate(f):
            if '\u2028' in line:
                skipped += 1
                continue
            line = line.rstrip().split()
            if len(line) != 2:
                skipped += 1
                continue
            assert len(line) == 2, (i, line)
            # assert line[0] not in word2id and line[1].isdigit(), (i, line)
            assert line[1].isdigit(), (i, line)
            if line[0] in word2id:
                skipped += 1
                print('%s already in vocab' % line[0])
                continue
            if not line[1].isdigit():
                skipped += 1
                print('Empty word at line %s with count %s' % (i, line))
                continue
            word2id[line[0]] = 4 + SPECIAL_WORDS + i - skipped  # shift because of extra words
            counts[line[0]] = int(line[1])
        f.close()
        id2word = {v: k for k, v in word2id.items()}
        dico = Dictionary(id2word, word2id, counts)
        logger.info("Read %i words from the vocabulary file." % len(dico))
        if skipped > 0:
            logger.warning("Skipped %i empty lines!" % skipped)
        return dico

    @staticmethod
    def index_data(path, dico):
        """
        Index sentences with a dictionary.
        """
        max_len = 100
        ids = []
        unk_words = {}

        # index sentences
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i % 1000000 == 0 and i > 0:
                    print(i)
                s = line.rstrip().split()
                # skip empty sentences
                if len(s) == 0:
                    print("Empty sentence in line %i." % i)
                # index sentence words
                count_unk = 0
                indexed = []
                for w in s:
                    word_id = dico.index(w, no_unk=False)
                    # if we find a special word which is not an unknown word, skip the sentence
                    if 0 <= word_id < 4 + SPECIAL_WORDS and word_id != 3:
                        logger.warning('Found unexpected special word "%s" (%i)!!' % (w, word_id))
                        continue
                    assert word_id >= 0
                    indexed.append(word_id)
                    if word_id == dico.unk_index:
                        unk_words[w] = unk_words.get(w, 0) + 1
                        count_unk += 1
                if len(indexed) >= max_len:
                    indexed = indexed[:max_len]
                indexed.extend([2] * (max_len - len(indexed)))
                assert len(indexed) == 100, 'sequence length is not equivalent'
                ids.append(indexed)
        return ids

def tag_encoding(tag_path):
    total_labels = []
    tags = set()

    with open(tag_path, mode='r', encoding='utf-8') as r2:
        for line in r2:
            line = line.rstrip('\n').split(' ')
            total_labels.append(line)
            for i in line:
                tags.add(i)

    import collections
    def flatten(x):
        result = []
        for el in x:
            if isinstance(x, collections.Iterable) and not isinstance(el, str):
                result.extend(flatten(el))
            else:
                result.append(el)
        return result

    import pandas as pd

    tags = [i for i in tags]
    tag_ids = range(len(tags))
    tag2id = pd.Series(tag_ids, index=tags)
    id2tag = pd.Series(tags, index=tag_ids)

    max_len = 100

    def Y_padding(tags):
        ids = list(tag2id[tags])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([tag2id['O']] * (max_len - len(ids)))
        return ids

    df_data = pd.DataFrame({'tags': total_labels}, index=range(len(total_labels)))
    df_data['y'] = df_data['tags'].apply(Y_padding)
    y = np.asarray(list(df_data['y'].values))
    return y, tag2id, id2tag


def text_encoding(model_path, txt_path, tag_path, pkl_path):
    '''
    index text to id and transfer to pkl format
    '''
    reloaded = torch.load(model_path)
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])

    ids = Dictionary.index_data(txt_path, dico)
    import numpy as np
    x = np.asarray(ids)          # ids包含每行的固定长度的词典id表示 size:(num_seq, max_len)
    y, tag2id, id2tag = tag_encoding(tag_path)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1024)
    with open(pkl_path, 'wb') as outp:
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
    logger.info('Finished saving the data in training_data.pkl')


def array2tensor(x_train, x_test, y_train, y_test):
    '''
    transform nparray and list to tensor
    '''
    x_train_tensor = torch.from_numpy(x_train)
    x_test_tensor = torch.from_numpy(x_test)
    y_train_tensor = torch.from_numpy(np.array(y_train))
    y_test_tensor = torch.from_numpy(np.array(y_test))
    x_train_tensor = x_train_tensor.float()
    y_train_tensor = y_train_tensor.float()
    return x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor

def generate_DataLoader(X_train, X_test, Y_train, Y_test, batch_size):
    X_train, X_test, Y_train, Y_test = array2tensor(X_train, X_test, Y_train, Y_test)
    train_dataset = Data.TensorDataset(X_train, Y_train)
    dev_dataset = Data.TensorDataset(X_test, Y_test)
    train_iter = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    dev_iter = Data.DataLoader(
        dataset=dev_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return train_iter, dev_iter

if __name__ == '__main__':
    model_path = '/Users/pengjie/Documents/code/XLM/model/mlm_tlm_xnli15_1024.pth'
    txt_path = '/Users/pengjie/Documents/code/XLM/model/dataset_seg/remove_bpe.txt'
    tag_path = '/Users/pengjie/Documents/code/XLM/model/dataset_seg/tag_bpe.txt'
    pkl_path = '/Users/pengjie/Documents/code/XLM/model/dataset_seg/train/train.pkl'
    text_encoding(model_path, txt_path, tag_path, pkl_path)