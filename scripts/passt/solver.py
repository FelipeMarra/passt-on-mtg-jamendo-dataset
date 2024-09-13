import os
import time
import numpy as np
import datetime

import torch
import torch.nn as nn

from model import PaSSTMTG

class Solver(object):
    def __init__(self, data_loader, valid_loader, config):
        # Data loader
        self.data_loader = data_loader
        self.valid_loader = valid_loader

        # Training settings
        self.n_epochs = 10
        self.lr = 1e-4
        self.log_step = 50
        self.is_cuda = torch.cuda.is_available()
        self.model_save_path = config.model_save_path
        self.batch_size = config.batch_size
        self.tag_list = self.get_tag_list(config)
        if config.subset == 'all':
            self.num_class = 183
        elif config.subset == 'genre':
            self.num_class = 87
            self.tag_list = self.tag_list[:87]
        elif config.subset == 'instrument':
            self.num_class = 40
            self.tag_list = self.tag_list[87:127]
        elif config.subset == 'moodtheme':
            self.num_class = 56
            self.tag_list = self.tag_list[127:]
        elif config.subset == 'top50tags':
            self.num_class = 50
        self.model_fn = os.path.join(self.model_save_path, 'best_model.pth')
        self.accs_fn = 'accs'+config.subset+'_'+str(config.split)+'.npy'

        # Build model
        self.build_model()

    def build_model(self):
        # model and optimizer
        model = PaSSTMTG(n_classes=self.num_class)

        if self.is_cuda:
            self.model = model
            self.model.cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.lr)

    def load(self, filename):
        S = torch.load(filename)
        self.model.load_state_dict(S)

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({'model': model}, filename)

    def to_var(self, x):
        if self.is_cuda:
            x = x.cuda()
        return x

    def train(self):
        start_t = time.time()
        best_accs_mean = 0
        reconst_loss = nn.BCELoss()

        for epoch in range(self.n_epochs):
            # train
            self.model.train()
            ctr = 0
            for x, y, _ in self.data_loader:
                ctr += 1

                # variables to cuda
                x = self.to_var(x)
                y = self.to_var(y)

                # predict
                out = self.model(x)
                loss = reconst_loss(out, y)

                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print log
                if (ctr) % self.log_step == 0:
                    print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            epoch+1, self.n_epochs, ctr, len(self.data_loader), loss.item(),
                            datetime.timedelta(seconds=time.time()-start_t)))

            # validation
            accs_mean = self._validation(start_t, epoch)

            # save model
            if accs_mean > best_accs_mean:
                print('best model acc mean: %4f' % accs_mean)
                best_accs_mean = accs_mean

            torch.save(self.model.state_dict(), os.path.join(self.model_save_path, f'model_iter_{ctr}_acc_{accs_mean}.pth'))

        print("[%s] Train finished. Elapsed: %s"
                % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.timedelta(seconds=time.time() - start_t)))

    def get_batch_acc(self, out, y):
        accuracies = []

        #for each prediction and ground truth in the batch
        for prd, gt in zip(out,y):
            prd = (prd > 0.5) * 1
            acc = ((prd == gt) * 1).float().mean()
            accuracies.append(acc)

        return accuracies

    def _validation(self, start_t, epoch):
        accuracies = []
        ctr = 0
        self.model.eval()
        reconst_loss = nn.BCELoss()
        for x, y, _ in self.valid_loader:
            ctr += 1

            # variables to cuda
            x = self.to_var(x)
            y = self.to_var(y)

            # predict
            out = self.model(x)
            loss = reconst_loss(out, y)

            # print log
            if (ctr) % self.log_step == 0:
                print("[%s] Epoch [%d/%d], Iter [%d/%d] valid loss: %.4f Elapsed: %s" %
                        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        epoch+1, self.n_epochs, ctr, len(self.valid_loader), loss.item(),
                        datetime.timedelta(seconds=time.time()-start_t)))

            # append prediction
            out = out.detach().cpu()
            y = y.detach().cpu()

            batch_accs = self.get_batch_acc(out, y)
            print(f"\n Validation\n all accs: {batch_accs} \n")
            for acc in batch_accs:
                accuracies.append(acc)

        return np.array(accuracies).mean()

    def get_tag_list(self, config):
        if config.subset == 'top50tags':
            path = 'tag_list_50.npy'
        else:
            path = 'tag_list.npy'
        tag_list = np.load(path)
        return tag_list

    def test(self):
        start_t = time.time()
        reconst_loss = nn.BCELoss()

        self.load(self.model_fn)
        self.model.eval()
        ctr = 0
        accuracies = []
        song_array = [] # song array
        for x, y, fn in self.data_loader:
            ctr += 1

            # variables to cuda
            x = self.to_var(x)
            y = self.to_var(y)

            # predict
            out = self.model(x)
            loss = reconst_loss(out, y)

            # print log
            if (ctr) % self.log_step == 0:
                print("[%s] Iter [%d/%d] test loss: %.4f Elapsed: %s" %
                        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        ctr, len(self.data_loader), loss.item(),
                        datetime.timedelta(seconds=time.time()-start_t)))

            # append prediction
            out = out.detach().cpu()
            y = y.detach().cpu()

            batch_accs = self.get_batch_acc(out, y)
            for acc in batch_accs:
                accuracies.append(acc)

            for song in fn:
                song_array.append(song)

        # save aucs
        np.save(open(self.accs_fn, 'wb'), accuracies)
        np.save(open('song_list.npy', 'wb'), song_array)
