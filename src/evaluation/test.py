import os
import torch

from src.utils import AttrDict
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.model.transformer import TransformerModel

model_path = '/Users/pengjie/Documents/code/XLM/model/mlm_tlm_xnli15_1024.pth'
reloaded = torch.load(model_path)
params = AttrDict(reloaded['params'])

# build dictionary / update parameters
dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
params.n_words = len(dico)
params.bos_index = dico.index(BOS_WORD)
params.eos_index = dico.index(EOS_WORD)
params.pad_index = dico.index(PAD_WORD)
params.unk_index = dico.index(UNK_WORD)
params.mask_index = dico.index(MASK_WORD)

# build model / reload weights
model = TransformerModel(params, dico, True, True)
model.eval()
model.load_state_dict(reloaded['model'])

# list of (sentences, lang)
sentences = [
    ('the following secon@@ dary charac@@ ters also appear in the nov@@ el .', 'en'),
    ('les zones rurales offr@@ ent de petites routes , a deux voies .', 'fr'),
    ('luego del cri@@ quet , esta el futbol , el sur@@ f , entre otros .', 'es'),
    ('am 18. august 1997 wurde der astero@@ id ( 76@@ 55 ) adam@@ ries nach ihm benannt .', 'de'),
    ('اصدرت عدة افلام وث@@ اي@@ قية عن حياة السيدة في@@ روز من بينها :', 'ar'),
    ('此外 ， 松@@ 嫩 平原 上 还有 许多 小 湖泊 ， 当地 俗@@ 称 为 “ 泡@@ 子 ” 。', 'zh'),
]

# add </s> sentence delimiters
sentences = [(('</s> %s </s>' % sent.strip()).split(), lang) for sent, lang in sentences]

bs = len(sentences)
slen = max([len(sent) for sent, _ in sentences])

word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
for i in range(len(sentences)):
    sent = torch.LongTensor([dico.index(w) for w in sentences[i][0]])
    word_ids[:len(sent), i] = sent

lengths = torch.LongTensor([len(sent) for sent, _ in sentences])
langs = torch.LongTensor([params.lang2id[lang] for _, lang in sentences]).unsqueeze(0).expand(slen, bs) if params.n_langs > 1 else None

tensor = model('fwd', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()
print(tensor.size())
print(tensor[0])
