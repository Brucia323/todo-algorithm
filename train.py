from datetime import datetime

import torch
from matplotlib.pyplot import cla, legend, plot, show
from numpy import array
from numpy.core.fromnumeric import mean, squeeze, std
from pandas import read_csv
from pandas.core.frame import DataFrame
from torch import unsqueeze
from torch.functional import Tensor
from torch.nn import Module
from torch.nn.modules.container import Sequential
from torch.nn.modules.linear import Linear
from torch.nn.modules.loss import MSELoss
from torch.nn.modules.rnn import LSTM
from torch.optim import Adam
from torch.serialization import save
from torch.utils.data import DataLoader, Dataset


def generate_df_affect_by_n_days(series, n, index=False):
    if len(series) <= n:
        return Exception(
            "The Length of series is %d,while affect by (n=%d)." % (len(series), n)
        )
    df = DataFrame()
    for i in range(n):
        df["c%d" % i] = series.tolist()[i : -(n - i)]
    df["y"] = series.tolist()[n:]
    if index:
        df.index = series.index[n:]
    return df


def read_data(column="amount", n=30, all_too=True, index=False, train_end=-300):
    df = read_csv("000001.csv", index_col=0, encoding="gbk")
    df.index = list(map(lambda x: datetime.strptime(x, "%Y/%m/%d"), df.index))
    df_column = df[column].copy()
    df_column_train, df_column_test = df_column[:train_end], df_column[train_end - n :]
    df_generate_from_df_culumn_train = generate_df_affect_by_n_days(
        df_column_train, n, index=index
    )
    if all_too:
        return df_generate_from_df_culumn_train, df_column, df.index.tolist()
    return df_generate_from_df_culumn_train


class RNN(Module):
    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.rnn = LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = Sequential(Linear(64, 1))

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out)
        return out


class TrainSet(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


n = 30
LR = 0.001
EPOCH = 500
train_end = -100

df, df_all, df_index = read_data("amount", n=n, train_end=train_end)

df_all = array(df_all.tolist())
plot(df_index, df_all, label="real-data")
df_numpy = array(df)

df_numpy_mean = mean(df_numpy)
df_numpy_std = std(df_numpy)

df_numpy = (df_numpy - df_numpy_mean) / df_numpy_std
df_tensor = Tensor(df_numpy).cuda()

trainset = TrainSet(df_tensor)
trainloader = DataLoader(trainset, batch_size=512, shuffle=True)

rnn = RNN(n).cuda()
optimizer = Adam(rnn.parameters(), lr=LR)
loss_func = MSELoss()

for step in range(EPOCH):
    for tx, ty in trainloader:
        output = rnn(unsqueeze(tx, dim=0))
        loss = loss_func(squeeze(output), ty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(step, loss)
    if step % 10:
        save(rnn, "rnn.pk1")
save(rnn, "rnn.pk1")

generate_data_train = []
generate_data_test = []

test_index = len(df_all) + train_end

df_all_normal = (df_all - df_numpy_mean) / df_numpy_std
df_all_normal_tensor = Tensor(df_all_normal).cuda()
for i in range(n, len(df_all)):
    x = df_all_normal_tensor[i - n : i]
    x = unsqueeze(unsqueeze(x, dim=0), dim=0)
    y = rnn(x)
    if i < test_index:
        generate_data_train.append(
            squeeze(y).cpu().detach().numpy() * df_numpy_std + df_numpy_mean
        )
    else:
        generate_data_test.append(
            squeeze(y).cpu().detach().numpy() * df_numpy_std + df_numpy_mean
        )
plot(df_index[n:train_end], generate_data_train, label="generate-train")
plot(df_index[train_end:], generate_data_test, label="generate-test")
legend()
show()
cla()
plot(df_index[train_end:], df_all[train_end:], label="real-data")
plot(df_index[train_end:], generate_data_test[:], label="generate-test")
legend()
show()
