import pandas as pd
import numpy as np
import pendulum
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import random
import os
import matplotlib.pyplot as plt
from datetime import datetime

from src.time_handle import *
from src import utils
from src.features.scale_dataset import scale_periods
from src.features.denoise_dataset import denoise_periods
from subrepos.DeepLearning_Financial.models import Autoencoder
from subrepos.DeepLearning_Financial.models import Sequence
from subrepos.DeepLearning_Financial.utils import prepare_data_lstm, ExampleDataset, evaluate_lstm, backtest, save_checkpoint

# ---------------------------------------------- Data Prepare ----------------------------------------------
xlsx = pd.ExcelFile('./data/interim/clean_data.xlsx')

sheet_names = ['csi300 index data', 'nifty 50 index data', 'hangseng index data', 'nikkei 225 index data', 's&p500 index data', 'djia index data']
data_names = ['csi300', 'nifty50', 'hangseng', 'nikkei225', 'sp500', 'djia']
data_dfs = list(map(lambda x: pd.read_excel(xlsx, x), sheet_names))

data_dict = dict(zip(data_names, data_dfs))

# data_dict = change_date_to_datetime(data_dict)

# -----------------------------------------------------------------------------------------------------------

# -------------------------------------------- Data Preprocessing -------------------------------------------
# plan to fix code to put these 3 parameters as input to calculate split periods.
train_month = 24
val_month = 3
test_month = 3

#for k, v in d.items():
#    exec('%s = %s' % (k, v))
#dt.datetime(2007,6,1)+relativedelta(years=2)

###### val & test data will be scaled with parameters of train data

###### data_dict -> period_split -> tvt_split -> DWT -> scale


interval_data_dict = utils.dict_interval_split(data_dict)

tvt_interval_data_dict = utils.dd_tvt_split(interval_data_dict)

# Original
final_original_data_dict = scale_periods(tvt_interval_data_dict)

# Denoise
denoised_tvt_interval_data_dict = denoise_periods(tvt_interval_data_dict)
final_denoised_data_dict = scale_periods(denoised_tvt_interval_data_dict)

# gpu settings
gpu_id = '0'
device = torch.device("cuda:"+gpu_id if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------------------------------------------------


for index_name, index_dict in final_denoised_data_dict.items(): # train and predict per each index

    y_compounded_test_pred_lst = []
    y_compounded_test_target_lst = []
    date_label_lst = []

    for roll_num, roll_index in index_dict.items(): # train and predict with each rolled data

        print('{} {} period START'.format(index_name, roll_num))
        # ------------------------------------------- Stacked Auto Encoder ------------------------------------------

        denoised_train_x = roll_index[1].copy()
        denoised_val_x = roll_index[2].copy()
        denoised_test_x = roll_index[3].copy()

        original_train_x = final_original_data_dict[index_name][roll_num][1].copy()
        original_val_x = final_original_data_dict[index_name][roll_num][2].copy()
        original_test_x = final_original_data_dict[index_name][roll_num][3].copy()

        num_hidden_1 = 10
        num_hidden_2 = 10
        num_hidden_3 = 10
        num_hidden_4 = 10

        n_epoch = 2500  # 20000

        if roll_num == 1:
            auto1 = Autoencoder(denoised_train_x.shape[1], num_hidden_1)
        auto1.cuda()
        auto1.fit(torch.tensor(denoised_train_x.values).cuda(), torch.tensor(denoised_val_x.values).cuda(), n_epoch=n_epoch)

        train_inputs = torch.autograd.Variable(torch.from_numpy(denoised_train_x.values.astype(np.float32)))
        val_inputs = torch.autograd.Variable(torch.from_numpy(denoised_val_x.values.astype(np.float32)))

        if roll_num == 1:
            auto2 = Autoencoder(num_hidden_1, num_hidden_2)
        auto2.cuda()
        train_auto1_out = auto1.encoder(train_inputs).data.numpy()
        val_auto1_out = auto1.encoder(val_inputs).data.numpy()
        auto2.fit(train_auto1_out, val_auto1_out, n_epoch=n_epoch)

        if roll_num == 1:
            auto3 = Autoencoder(num_hidden_2, num_hidden_3)
        auto3.cuda()
        train_auto1_out = torch.autograd.Variable(torch.from_numpy(train_auto1_out.astype(np.float32)))
        val_auto1_out = torch.autograd.Variable(torch.from_numpy(val_auto1_out.astype(np.float32)))
        train_auto2_out = auto2.encoder(train_auto1_out).data.numpy()
        val_auto2_out = auto2.encoder(val_auto1_out).data.numpy()
        auto3.fit(train_auto2_out, val_auto2_out, n_epoch=n_epoch)

        if roll_num == 1:
            auto4 = Autoencoder(num_hidden_3, num_hidden_4)
        auto4.cuda()
        train_auto2_out = torch.autograd.Variable(torch.from_numpy(train_auto2_out.astype(np.float32)))
        val_auto2_out = torch.autograd.Variable(torch.from_numpy(val_auto2_out.astype(np.float32)))
        train_auto3_out = auto3.encoder(train_auto2_out).data.numpy()
        val_auto3_out = auto3.encoder(val_auto2_out).data.numpy()
        auto4.fit(train_auto3_out, val_auto3_out, n_epoch=n_epoch)

        # Change to evaluation mode, in this mode the network behaves differently, e.g. dropout is switched off and so on
        auto1.eval()
        auto2.eval()
        auto3.eval()
        auto4.eval()

        X_train = denoised_train_x.values.copy()
        X_train = torch.autograd.Variable(torch.from_numpy(X_train.astype(np.float32)))
        train_encoded = auto4.encoder(auto3.encoder(auto2.encoder(auto1.encoder(X_train))))
        train_encoded = train_encoded.data.numpy()

        # ---- encode validation and test data using autoencoder trained only on training data
        X_validate = denoised_val_x.values.copy()
        X_validate = torch.autograd.Variable(torch.from_numpy(X_validate.astype(np.float32)))
        validate_encoded = auto4.encoder(auto3.encoder(auto2.encoder(auto1.encoder(X_validate))))
        validate_encoded = validate_encoded.data.numpy()

        X_test = denoised_test_x.values
        X_test = torch.autograd.Variable(torch.from_numpy(X_test.astype(np.float32)))
        test_encoded = auto4.encoder(auto3.encoder(auto2.encoder(auto1.encoder(X_test))))
        test_encoded = test_encoded.data.numpy()

        # switch back to training mode
        auto1.train()
        auto2.train()
        auto3.train()
        auto4.train()

        # -----------------------------------------------------------------------------------------------------------

        # --------------------------------------- Return prediction using LSTM --------------------------------------

        # Prepare data for LSTM model

        # split the entire training time-series into pieces, depending on the number
        # of time steps for the LSTM

        time_steps = 4
        train_logreturn = True

        data_close = pd.concat([final_original_data_dict[index_name][roll_num]['raw_train_close'].copy(),
                                final_original_data_dict[index_name][roll_num]['raw_val_close'].copy(),
                                final_original_data_dict[index_name][roll_num]['raw_test_close'].copy()]).reset_index(drop=True)

        args = (train_encoded, validate_encoded, test_encoded)

        x_concat = np.concatenate(args)

        validate_encoded_extra = np.concatenate((train_encoded[-time_steps:], validate_encoded))
        test_encoded_extra = np.concatenate((validate_encoded[-time_steps:], test_encoded))

        y_train_input = data_close[:-len(validate_encoded) - len(test_encoded)]
        y_val_input = data_close[-len(test_encoded) - len(validate_encoded) - 1:-len(test_encoded)]
        y_test_input = data_close[-len(test_encoded) - 1:]

        x, y = prepare_data_lstm(train_encoded, y_train_input, time_steps, log_return=train_logreturn, train=True)
        x_v, y_v = prepare_data_lstm(validate_encoded_extra, y_val_input, time_steps, log_return=False, train=False)
        x_te, y_te = prepare_data_lstm(test_encoded_extra, y_test_input, time_steps, log_return=False, train=False)

        x_test = x_te
        x_validate = x_v
        x_train = x

        y_test = y_te
        y_validate = y_v
        y_train = y

        y_train = y_train.values

        # LSTM model

        batchsize = 60

        trainloader = ExampleDataset(x_train, y_train, batchsize)
        valloader = ExampleDataset(x_validate, y_validate, 1)
        testloader = ExampleDataset(x_test, y_test, 1)

        # set ramdom seed to 0
        np.random.seed(0)
        torch.manual_seed(0)

        # build the model
        if roll_num == 1:
            seq = Sequence(num_hidden_4, hidden_size=100, nb_layers=3)
        seq.cuda()

        resume = ""

        # if a path is given in resume, we resume from a checkpoint
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            seq.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

        # get the number of model parameters
        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in seq.parameters()])))

        # we use the mean squared error loss
        criterion = nn.MSELoss()

        optimizer = optim.Adam(params=seq.parameters(), lr=0.005)

        start_epoch = 1
        epochs = 1000  # 5000

        global_loss_val = np.inf
        # begin to train
        global_profit_val = -np.inf

        # rescaling factor
        train_close_min = final_original_data_dict[index_name][roll_num]['min'][3]
        train_close_max = final_original_data_dict[index_name][roll_num]['max'][3]
        train_close_range = final_original_data_dict[index_name][roll_num]['range'][3]

        for i in range(start_epoch, epochs+1):
            seq.train()
            loss_train = 0

            # shuffle ONLY training set
            combined = list(zip(x_train, y_train))
            random.shuffle(combined)
            x_train = []
            y_train = []
            x_train[:], y_train[:] = zip(*combined)

            # initialize trainloader with newly shuffled training data
            trainloader = ExampleDataset(x_train, y_train, batchsize)

            pred_train = []
            target_train = []
            for j in range(len(trainloader)):
                sample = trainloader[j]
                sample_x = sample["x"]

                if len(sample_x) != 0:
                    sample_x = np.stack(sample_x)
                    input = Variable(torch.FloatTensor(sample_x), requires_grad=False)
                    input = torch.transpose(input, 0, 1)
                    target = Variable(torch.FloatTensor([x for x in sample["y"]]), requires_grad=False)

                    optimizer.zero_grad()
                    out = seq(input)
                    out = out.reshape(-1)
                    loss = criterion(out, target)

                    loss_train += float(loss.data.numpy())
                    pred_train.extend(out.data.numpy().flatten().tolist())
                    target_train.extend(target.data.numpy().flatten().tolist())

                    loss.backward()

                    optimizer.step()
            if i % 200 == 0:

                # Show Plot of TRAIN results of y(return) & prediction
                #plt.plot(pred_train)
                #plt.plot(target_train)
                #plt.show()

                loss_val, pred_val, target_val = evaluate_lstm(dataloader=valloader, model=seq, criterion=criterion)

                # Show Plot of VALIDATION results of y(return) & prediction
                #plt.scatter(range(len(pred_val)), pred_val)
                #plt.scatter(range(len(pred_val)), target_val)
                #plt.show()

                index, real = backtest(pred_val, y_validate)

                print(index[-1])                # save according to profitability
                if index[-1] > global_profit_val and i > 200:
                    print("CURRENT BEST")
                    global_profit_val = index[-1]
                    save_checkpoint({'epoch': i, 'state_dict': seq.state_dict()}, is_best=True,
                                    filename='checkpoint_lstm.pth.tar')
                else:
                    save_checkpoint({'epoch': i, 'state_dict': seq.state_dict()}, is_best=False,
                                    filename='checkpoint_lstm.pth.tar')

                print("LOSS TRAIN at {} epoch: ".format(i) + str(float(loss_train)))
                print("LOSS VAL at {} epoch: ".format(i) + str(float(loss_val)))
            if i % 50 == 0:
                print('{} epoch done'.format(i))

        # do the final test
        # first load the best checkpoint on the val set

        resume = "./runs/checkpoint/model_best.pth.tar"
        # resume = "./runs/HF/checkpoint_lstm.pth.tar"

        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            seq.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

        seq.eval()

        loss_test, preds_test, target_test = evaluate_lstm(dataloader=testloader, model=seq, criterion=criterion)

        print("LOSS TEST: " + str(float(loss_test)))

        print('{} {} period END'.format(index_name, roll_num))

        # -----------------------------------------------------------------------------------------------------------

        # ------------------------------------------- Plot & Save Results  ------------------------------------------

        # plot an 1 period result & compounded period result
        y_test_pred_lst = preds_test
        y_test_target_lst = target_test
        date_label = final_original_data_dict[index_name][roll_num][3].index

        #temp2 = y_test.values.flatten().tolist()
        y_compounded_test_pred_lst.extend(y_test_pred_lst)
        y_compounded_test_target_lst.extend(y_test_target_lst)
        date_label_lst.extend(date_label)



        # show just an 1 period return result
        plt.plot(date_label, y_test_pred_lst, 'bo-')
        plt.plot(date_label, y_test_target_lst, 'r*-')
        plt.title('{} return result at {} period'.format(index_name, roll_num))
        plt.savefig("./plot/{}_test_1pred".format(index_name) + str(roll_num) + ".png")
        plt.show()
        #plt.scatter(range(len(preds_test)), preds_test)
        # plt.scatter(range(len(y_test_lst)), y_test_lst)
        #plt.scatter(range(len(temp2)), temp2)

        # show compounded return result so far
        plt.plot(date_label_lst, y_compounded_test_pred_lst, 'bo-')
        plt.plot(date_label_lst, y_compounded_test_target_lst, 'r*-')
        plt.title('{} return result 1~{} period'.format(index_name, roll_num))
        plt.savefig("./plot/{}_test_preds".format(index_name) + str(roll_num) + ".png")
        plt.show()


        # 1 period backtest result
        test_index, test_real = backtest(preds_test, y_test)

        #
        plt.plot(date_label, test_index[1:], 'bo-')
        plt.plot(date_label, test_real[1:], 'r*-')
        plt.title('{} transaction backtest result at {} period'.format(index_name, roll_num))
        plt.savefig("./plot/{}_trans_backtest".format(index_name) + str(roll_num) + ".png")
        plt.show()

        if index_name == 'nifty50' and roll_num == 23:
            print('stop')
            print('stop')

    print(str(index_name) + 'Done' )




