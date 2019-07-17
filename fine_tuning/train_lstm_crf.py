# coding: utf-8
import os
import logging
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm import trange

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# loss_fn = CrossEntropyLoss()


def train(model, train_iter, dev_iter, params):
    for param in model.xlm.parameters():  ## freeze layers
        param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=0.003, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    iteration, best_f1 = 0, 0

    for epoch in trange(params.n_epochs):
        for sentence, tags in train_iter:
            model.train()
            iteration += 1
            optimizer.zero_grad()

            # sentence = torch.tensor(sentence, dtype=torch.long)
            sentence = sentence.long().transpose(0, 1).to(device)  # slen * bs
            # tags = torch.tensor([tag2id[t] for t in tags], dtype=torch.long)
            tags = tags.long().to(device)

            lengths = torch.LongTensor([params.max_len] * sentence.size(1)).to(device)
            # langs = ''
            # logits = model(sentence, lengths)
            loss = model.log_likelihood(sentence, lengths, tags)

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=2)

            optimizer.step()

            if iteration % 20 == 0:
                logging.info(
                    '\rEpoch[{}] - Iteration[{}] - loss: {}'.format(epoch, iteration, loss.item()))

            if iteration % 20 == 0:
                _, _, eval_f1 = eval(model, dev_iter, params)
                if eval_f1 > best_f1:
                    best_f1 = eval_f1
                    save(model, "./dumped", iteration)


def eval(model, dev_iter, params):
    model.eval()

    aver_loss = 0
    preds, labels = [], []
    for sentence, tags in dev_iter:
        sentence = sentence.long().transpose(0, 1).to(device)
        tags = tags.long().to(device)

        lengths = torch.LongTensor([params.max_len] * sentence.size(1)).to(device)
        pred = model(sentence, lengths)
        loss = model.log_likelihood(sentence, lengths, tags)
        aver_loss += loss.item()

        for i in pred:
            preds += i
        for i in tags.tolist():
            labels += i

    aver_loss /= (len(dev_iter) * params.batch_size)
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    report = classification_report(labels, preds)
    print(report)

    logging.info('\nEvaluation - loss: {:.6f}  precision: {:.4f}  recall: {:.4f}  f1: {:.4f} \n'.format(aver_loss,
                                                                                                        precision,
                                                                                                        recall, f1))
    return precision, recall, f1


def save(model, save_dir, iteration):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    path_name = os.path.join(save_dir, "model-" + str(iteration) + ".pkl")
    torch.save(model.state_dict(), path_name)
    logging.info("model has been saved in {}".format(path_name))
