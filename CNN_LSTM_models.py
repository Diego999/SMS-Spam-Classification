import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


# Yield successive n-sized chunks from l.
def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        res.append(l[i:i + n])
    return res


class KimCNN(nn.Module):
    def __init__(self, word_emb_matrix, emb_num=8429, emb_dim=300, output_channel=256, class_num=2, dropout=0.5):
        super(KimCNN, self).__init__()
        output_channel = output_channel
        target_class = class_num

        Ks = 3  # There are three conv net here
        input_channel = 1
        self.non_static_embed = nn.Embedding(emb_num, emb_dim)
        self.non_static_embed.weight.requires_grad = True
        self.non_static_embed.weight.data.copy_(torch.from_numpy(word_emb_matrix).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)) # It's a parameter of nn.Embedding

        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, emb_dim), padding=(2, 0))
        self.conv2 = nn.Conv2d(input_channel, output_channel, (4, emb_dim), padding=(3, 0))
        self.conv3 = nn.Conv2d(input_channel, output_channel, (5, emb_dim), padding=(4, 0))

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(Ks * output_channel, target_class)

    def forward(self, x, length):
        x = self.non_static_embed(x)
        x = x.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)

        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        # (batch, channel_output, ~=sent_len) * Ks
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        # (batch, channel_output) * Ks
        x = torch.cat(x, 1)  # (batch, channel_output * Ks)
        x = self.dropout(x)
        logit = self.fc1(x)  # (batch, target_size)
        return logit


class LSTM(nn.Module):
    def __init__(self, word_emb_matrix, emb_num=8429, emb_dim=300, hid_dim=256, class_num=2, dropout=0.5):
        super(LSTM, self).__init__()
        self.non_static_embed =nn.Embedding(emb_num, emb_dim)
        self.non_static_embed.weight.requires_grad = True
        self.non_static_embed.weight.data.copy_(torch.from_numpy(word_emb_matrix).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor))  # It's a parameter of nn.Embedding

        self.hid_dim = hid_dim
        self.rnn = nn.LSTM(emb_dim, hid_dim, dropout=dropout, bias=True)
        self.linear_model = nn.Linear(hid_dim * 1, class_num)

    def forward(self, x, length):
        x = nn.utils.rnn.pack_padded_sequence(self.non_static_embed(x), length, batch_first=True)
        _, hidden = self.rnn(x)
        h_pool, _ = torch.max(hidden[0], 0)
        output = self.linear_model(h_pool)
        return output


class CNN_LSTM_Wrapper():
    def __init__(self, type, word_emb_matrix, patience=10):
        assert type in ['CNN', 'LSTM']
        self.patience = patience
        self.type = type
        self.model = KimCNN(np.array(word_emb_matrix)) if type == 'CNN' else LSTM(np.array(word_emb_matrix))

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3, weight_decay=0)
        self.criterion = nn.CrossEntropyLoss()

        self.best_model = None

    def fit(self, X, Y):
        val_size = int(len(X)*0.1)
        X_train, X_valid = X.tolist()[:-val_size], X.tolist()[-val_size:]
        Y_train, Y_valid = Y.tolist()[:-val_size], Y.tolist()[-val_size:]

        cost_val = []
        bad_counter = 0
        best_valid_loss = None
        self.format_model_saver = self.type + '_model_{0:05d}.ckpt'
        best_model_epoch = -1

        for epoch in range(1000):
            # Training
            self.model.train(mode=True)

            for X_batch, Y_batch in zip(chunks(X_train, 256), chunks(Y_train, 256)):
                length = [[i for i, xx in enumerate(x) if xx == 0] for x in X_batch]
                length = [l[0] if len(l) > 0 else len(X_batch[i]) for i, l in enumerate(length)]
                temp = sorted(zip(X_batch, Y_batch, length), key=lambda x:x[-1], reverse=True)
                X_batch = [x for x, _, _ in temp]
                Y_batch = [y for _, y, _ in temp]
                length = [l for _, _, l in temp]

                if not torch.cuda.is_available():
                    X_batch = Variable(torch.LongTensor(np.array(X_batch)), requires_grad=False)
                    Y_batch = Variable(torch.LongTensor(np.array(Y_batch)), requires_grad=False)
                    length = Variable(torch.LongTensor(np.array(length)), requires_grad=False)
                else:
                    X_batch = Variable(torch.cuda.LongTensor(np.array(X_batch)), requires_grad=False)
                    Y_batch = Variable(torch.cuda.LongTensor(np.array(Y_batch)), requires_grad=False)
                    length = Variable(torch.LongTensor(np.array(length)), requires_grad=False)

                self.optimizer.zero_grad()
                scores = self.model(X_batch, length)

                loss = self.criterion(scores, Y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, self.model.parameters()), 1.0)
                self.optimizer.step()

            # Validation
            self.model.eval()

            valid_loss = 0
            for X_batch, Y_batch in zip(chunks(X_valid, 256), chunks(Y_valid, 256)):
                length = [[i for i, xx in enumerate(x) if xx == 0][0] for x in X_batch]
                temp = sorted(zip(X_batch, Y_batch, length), key=lambda x:x[-1], reverse=True)
                X_batch = [x for x, _, _ in temp]
                Y_batch = [y for _, y, _ in temp]
                length = [l for _, _, l in temp]

                if not torch.cuda.is_available():
                    X_batch = Variable(torch.LongTensor(np.array(X_batch)), requires_grad=False)
                    Y_batch = Variable(torch.LongTensor(np.array(Y_batch)), requires_grad=False)
                    length = Variable(torch.LongTensor(np.array(length)), requires_grad=False)
                else:
                    X_batch = Variable(torch.cuda.LongTensor(np.array(X_batch)), requires_grad=False)
                    Y_batch = Variable(torch.cuda.LongTensor(np.array(Y_batch)), requires_grad=False)
                    length = Variable(torch.LongTensor(np.array(length)), requires_grad=False)

                scores = self.model(X_batch, length)
                valid_loss += self.criterion(scores, Y_batch)
            cost_val.append(valid_loss)

            # Save model
            torch.save(self.model.state_dict(), os.path.join('out/model/', self.format_model_saver.format(epoch)))

            # Early stopping
            if epoch > self.patience:
                if cost_val[-1] < best_valid_loss:
                    bad_counter = 0
                    best_valid_loss = cost_val[-1]
                    best_model_epoch = epoch
                else:
                    bad_counter += 1
            else:
                if len(cost_val) > 1 or len(cost_val) == 1:
                    best_valid_loss = cost_val[-1]
                    best_model_epoch = epoch

            print('{}\'th epoch'.format(epoch), '{:.5f}'.format(valid_loss.data[0]), 'Patience:{}'.format(bad_counter))

            if bad_counter >= self.patience:
                print("\n\nEarly stopping. Best model at Epoch " + '%04d' % (best_model_epoch))
                break

            # Delete other models to save disk space
            files = os.listdir('out/model/')
            for file in files:
                if not file.endswith('.txt') and self.type in file and '.' in file and '_' in file and int(file[file.rfind('/') + 1:].split('.')[0].split('_')[2]) != best_model_epoch:
                    os.remove(os.path.join('out/model/', file))

        self.best_model_epoch = best_model_epoch

    def predict(self, X):
        self.model.load_state_dict(torch.load(os.path.join('out/model/', self.format_model_saver.format(self.best_model_epoch))))

        # Testing
        self.model.eval()

        Y_hat = []
        for X_batch in chunks(X, 256):
            length = [[i for i, xx in enumerate(x) if xx == 0] for x in X_batch]
            length = [l[0] if len(l) > 0 else len(X_batch[i]) for i, l in enumerate(length)]
            length = [(i,l) for i,l in enumerate(length)]

            temp = sorted(zip(X_batch, length), key=lambda x: x[-1][-1], reverse=True)
            X_batch = [x for x, _ in temp]
            idx = np.argsort([l[0] for _, l in temp])
            length = [l[1] for _, l in temp]

            if not torch.cuda.is_available():
                X_batch = Variable(torch.LongTensor(np.array(X_batch)), requires_grad=False)
                length = Variable(torch.LongTensor(np.array(length)), requires_grad=False)
            else:
                X_batch = Variable(torch.cuda.LongTensor(np.array(X_batch)), requires_grad=False)
                length = Variable(torch.LongTensor(np.array(length)), requires_grad=False)

            y_hat = self.model(X_batch, length).data
            y_hat = np.array(y_hat)[idx]
            Y_hat.append(y_hat.tolist())

        return np.argmax(np.concatenate(Y_hat, axis=0), axis=1)
