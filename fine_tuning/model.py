import torch
from torch import nn
from src.model.transformer import TransformerModel
from torchcrf import CRF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class XLMForTokenClassification(nn.Module):
    def __init__(self, config, num_labels, params, dico, reloaded):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.xlm = TransformerModel(params, dico, True, True)
        self.xlm.eval()
        self.xlm.load_state_dict(reloaded['model'])

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(1024, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, word_ids, lengths, langs=None, causal=False):
        sequence_output = self.xlm('fwd', x=word_ids, lengths=lengths, causal=False).contiguous()
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



class XLM_BiLSTM_CRF(nn.Module):

    def __init__(self, config, num_labels, params, dico, reloaded):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.batch_size = config.batch_size
        self.hidden_dim = config.hidden_dim

        self.xlm = TransformerModel(params, dico, True, True)
        self.xlm.eval()
        self.xlm.load_state_dict(reloaded['model'])

        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_dim, config.num_class)
        self.apply(self.init_bert_weights)
        self.crf = CRF(config.num_class)

    def forward(self, word_ids, lengths, langs=None, causal=False):
        sequence_output = self.xlm('fwd', x=word_ids, lengths=lengths, causal=False).contiguous()
        sequence_output, _ = self.lstm(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return self.crf.decode(logits)

    def log_likelihood(self, word_ids, lengths, tags):
        sequence_output = self.xlm('fwd', x=word_ids, lengths=lengths, causal=False).contiguous()
        sequence_output, _ = self.lstm(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return - self.crf(logits, tags.transpose(0,1))

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()