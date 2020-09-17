import pandas as pd
import numpy as np
import sklearn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class Autoencoder(torch.nn.Module):
    def __init__(self, n_in, n_hidden=10, sparsity_target=0.05, sparsity_weight=0.2, lr=0.001,
                 weight_decay=0.0):  # lr=0.0001):
        super(Autoencoder, self).__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.weight_decay = weight_decay
        self.lr = lr
        self.build_model()

    # end constructor

    def build_model(self):
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.n_in, self.n_hidden),
            torch.nn.Sigmoid())
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.n_hidden, self.n_in))  # ,
        # torch.nn.Sigmoid())
        self.l1_loss = torch.nn.L1Loss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)

    # end method

    def forward(self, inputs):
        hidden = self.encoder(inputs)
        hidden_mean = torch.mean(hidden, dim=0)
        sparsity_loss = torch.sum(self.kl_divergence(self.sparsity_target, hidden_mean))
        return self.decoder(hidden), sparsity_loss

    # end method

    def kl_divergence(self, p, q):
        return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))  # Kullback Leibler divergence

    # end method

    def fit(self, X_train, X_val, n_epoch=10, batch_size=64, use_gpu=False, en_shuffle=True):
        for epoch in range(n_epoch):
            if en_shuffle:
                if epoch % 1000 == 0:
                    print("Data Shuffled")
                X_train = sklearn.utils.shuffle(X_train)
            for local_step, X_batch in enumerate(self.gen_batch(X_train, batch_size)):

                if use_gpu == True:
                    if type(X_batch) == torch.Tensor:
                        train_inputs = torch.autograd.Variable(X_batch.clone().detach().cuda())
                    else:
                        train_inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.float32)).cuda())
                else:
                    train_inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.float32)))

                train_outputs, train_sparsity_loss = self.forward(train_inputs)

                train_l1_loss = self.l1_loss(train_outputs, train_inputs)
                train_loss = train_l1_loss + self.sparsity_weight * train_sparsity_loss
                self.optimizer.zero_grad()  # clear gradients for this training step
                train_loss.backward()  # backpropagation, compute gradients
                self.optimizer.step()  # apply gradients
                if local_step != 0 and local_step % 7 == 0 and epoch % 1000 == 0:
                    if use_gpu == True:
                        if type(X_batch) == torch.Tensor:
                            val_inputs = torch.autograd.Variable(X_val.clone().detach().cuda())
                        else:
                            val_inputs = torch.autograd.Variable(torch.from_numpy(X_val.astype(np.float32)).cuda())

                    else:
                        val_inputs = torch.autograd.Variable(torch.from_numpy(X_val.astype(np.float32)))

                    val_outputs, val_sparsity_loss = self.forward(val_inputs)
                    val_l1_loss = self.l1_loss(val_outputs, val_inputs)
                    val_loss = val_l1_loss + self.sparsity_weight * val_sparsity_loss

                    print("Epoch %d/%d | Step %d/%d | train loss: %.4f | l1 loss: %.4f | sparsity loss: %.4f \n"
                          "val loss: %.4f | val l1 loss: %.4f | val sparsity loss: %.4f"
                          % (epoch + 1, n_epoch, local_step, len(X_train) // batch_size,
                             train_loss.item(), train_l1_loss.item(), train_sparsity_loss.item(),
                             val_loss.item(), val_l1_loss.item(), val_sparsity_loss.item()))

    # end method

    def gen_batch(self, df, batch_size):
        for i in range(0, len(df), batch_size):
            yield df[i: i + batch_size]
