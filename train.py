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
    '''给定时间序列输入，此函数生成一个 pandas DataFrame，其中列表示偏移 n 天的时间序列和一个目标列。
    
    Parameters
    ----------
    series
        包含用于生成 DataFrame 的数据的 pandas Series
    n
        用于生成数据帧的天数。
    index, optional
        一个布尔参数，用于确定是否将返回的 DataFrame 的索引设置为与输入系列的索引相同。如果为 True，则返回的 DataFrame 的索引将设置为从第 (n+1)
    个元素开始的输入系列的索引。如果为假，则
    
    Returns
    -------
        包含列“c0”到“cn-1”（其中 n 是参数 n 的值）和列“y”的 pandas DataFrame 对象。 “c”列包含输入序列中偏移 0 到 n-1 天的值，而“y”列包含输入序列中偏移 n
    天的值。如果参数索引
    
    '''
    # 此代码块检查输入序列的长度是否小于或等于 n 的值。如果是，它会引发异常，并显示一条消息，表明系列的长度不足以生成所需的 DataFrame。
    if len(series) <= n:
        return Exception(
            "The Length of series is %d,while affect by (n=%d)." % (len(series), n)
        )
    # `df = DataFrame()` 正在创建一个空的 pandas DataFrame 对象。该对象将用于在 `generate_df_affect_by_n_days`
    # 函数中存储生成的数据。
    df = DataFrame()
    # 此代码块在 pandas DataFrame 对象中生成列“c0”到“cn-1”（其中 n 是参数 n 的值）。 “c”列包含偏移 0 到 n-1 天的输入序列的值，而“y”列包含偏移 n
    # 天的输入序列的值。循环遍历 n 的范围并通过将输入序列从 i 切片到 -(n-i) 并使用字符串格式将其分配给 DataFrame 中的相应列来生成列，以生成列名称。
    for i in range(n):
        df["c%d" % i] = series.tolist()[i : -(n - i)]
    # `df["y"] = series.tolist()[n:]` 正在 pandas DataFrame `df` 中创建一个名为“y”的新列，并从第 n
    # 个开始为其分配输入序列的值指数。这样做是为了创建一个目标列，供机器学习模型进行预测。输入序列移动“n”天，以在输入特征和目标变量之间产生时间滞后。
    df["y"] = series.tolist()[n:]
    # `if index: df.index = series.index[n:]` 将生成的 pandas DataFrame 的索引设置为与输入系列相同，前提是 `index` 参数设置为
    # `True`。具体来说，它将生成的 DataFrame 的索引设置为与从第 (n+1) 个元素开始的输入系列相同。如果 `index` 参数设置为 `False`，则生成的 DataFrame
    # 的索引将是默认的整数索引。
    if index:
        df.index = series.index[n:]
    # `return df` 返回一个 pandas DataFrame 对象，其中包含“c0”到“cn-1”列（其中 n 是参数 n 的值）和一个列“y”。 “c”列包含偏移 0 到 n-1
    # 天的输入序列的值，而“y”列包含偏移 n 天的输入序列的值。此 DataFrame 由函数 generate_df_affect_by_n_days 生成，用作机器学习模型的输入数据。
    return df

def read_data(column="amount", n=30, all_too=True, index=False, train_end=-300):
    '''该函数从CSV文件中读取数据，根据指定的列和天数生成新的DataFrame，并将生成的DataFrame连同原始DataFrame和索引一起返回。
    
    Parameters
    ----------
    column, optional
        用于生成数据的数据集的列。
    n, optional
        生成数据的天数。
    all_too, optional
        一个布尔参数，用于确定是返回所有生成的数据还是仅返回从训练集中生成的数据。
    index, optional
        一个布尔参数，用于确定生成的 DataFrame 是否应具有索引。如果设置为 True，生成的 DataFrame 将有一个索引。如果设置为 False，生成的 DataFrame 将没有索引。
    train_end
        训练数据结束的索引位置。该索引位置之前的所有数据将用于训练，该索引位置之后的所有数据将用于测试。默认情况下，最后 300 个数据点用于测试。
    
    Returns
    -------
        包含基于指定列和天数生成的数据框、原始数据框列和原始数据框索引列表的元组。返回的具体元素取决于输入参数。
    
    '''
    # 这行代码正在读取名为“000001.csv”的 CSV 文件，并将其内容存储在名为“df”的 pandas DataFrame 对象中。 `index_col=0` 参数指定 CSV
    # 文件的第一列应该用作 DataFrame 的索引。 `encoding="gbk"` 参数指定 CSV 文件的字符编码。
    df = read_csv("000001.csv", index_col=0, encoding="gbk")
    # 这行代码将 pandas DataFrame `df` 的索引从字符串格式 ("%Y/%m/%d") 转换为日期时间格式。它使用 `map()` 函数将
    # `datetime.strptime()` 函数应用于索引的每个元素，从而将字符串转换为日期时间对象。然后使用 df.index 属性将生成的日期时间对象列表分配回 DataFrame
    # 的索引。
    df.index = list(map(lambda x: datetime.strptime(x, "%Y/%m/%d"), df.index))
    # `df_column = df[column].copy()` 正在从原始 DataFrame `df` 中创建由 `column` 参数指定的列的副本，并将其分配给新变量
    # `df_column`。这样做是为了确保对 df_column 所做的任何更改都不会影响原始 DataFrame df。
    df_column = df[column].copy()
    # `df_column_train` 和 `df_column_test` 是原始 `df_column` 系列的两个子集。 `df_column_train` 包含 `df_column`
    # 的值直到 `train_end` 索引，而 `df_column_test` 包含 `df_column` 的值从 `train_end - n`
    # 索引到序列的末尾。这用于将数据拆分为机器学习模型的训练和测试集。
    df_column_train, df_column_test = df_column[:train_end], df_column[train_end - n :]
    # `df_generate_from_df_culumn_train` 正在生成一个包含列“c0”到“cn-1”（其中 n 是参数 n 的值）和列“y”的 pandas DataFrame。
    # “c”列包含输入序列偏移 0 到 n-1 天的值，而“y”列包含输入序列偏移 n 天的值。此 DataFrame 由函数 generate_df_affect_by_n_days
    # 生成，用作机器学习模型的输入数据。 `index` 参数决定生成的 DataFrame 是否应该有索引。如果 `index` 设置为 `True`，生成的 DataFrame
    # 将有一个与输入序列匹配的索引，从第 (n+1) 个元素开始。如果 `index` 设置为 `False`，生成的 DataFrame 将有一个默认的整数索引。
    df_generate_from_df_culumn_train = generate_df_affect_by_n_days(
        df_column_train, n, index=index
    )
    # 此代码块正在检查 `all_too` 参数是否设置为 `True`。如果是，该函数返回一个包含三个元素的元组：从指定列和天数生成的 DataFrame
    # (`df_generate_from_df_culumn_train`)、原始 DataFrame 列 (`df_column`) 和原始 DataFrame 索引列表 (`
    # df.index.tolist()`）。如果 `all_too` 设置为 `False`，该函数仅返回从指定列和天数生成的 DataFrame
    # (`df_generate_from_df_culumn_train`)。
    if all_too:
        return df_generate_from_df_culumn_train, df_column, df.index.tolist()
    # `return df_generate_from_df_culumn_train` 返回一个 pandas DataFrame 对象，其中包含列“c0”到“cn-1”（其中 n 是参数 n
    # 的值）和一列“y”。 “c”列包含输入序列偏移 0 到 n-1 天的值，而“y”列包含输入序列偏移 n 天的值。此 DataFrame 由函数
    # generate_df_affect_by_n_days 生成，用作机器学习模型的输入数据。
    return df_generate_from_df_culumn_train


# 这是具有单个 LSTM 层和线性输出层的循环神经网络 (RNN) 的类。
class RNN(Module):
    def __init__(self, input_size):
        '''这是具有单个 LSTM 层和线性输出层的循环神经网络 (RNN) 的初始化函数。
        
        Parameters
        ----------
        input_size
            RNN 输入的大小。这是每个时间步的输入数据中的特征数。
        
        '''
        # `super(RNN, self).__init__()` 调用的是`RNN` 的父类，也就是`Module` 的`__init__()` 方法。这是在初始化特定于 RNN
        # 类的属性之前正确初始化 `Module` 类及其属性所必需的。
        super(RNN, self).__init__()
        # 这行代码正在创建一个具有指定输入大小、隐藏大小、层数和批处理第一个参数的 LSTM 层。 LSTM 层是一种常用于序列建模任务的递归神经网络
        # (RNN)。输入大小参数指定输入数据中预期特征的数量，而隐藏大小参数指定 LSTM 隐藏状态中的特征数量。 num_layers 参数指定 LSTM
        # 中的循环层数，batch_first 参数指定输入和输出张量是否应将批量大小作为第一维。
        self.rnn = LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        # `self.out = Sequential(Linear(64, 1))` 正在创建一个输入大小为 64，输出大小为 1 的线性层，然后将其包装在顺序容器中。这用作 RNN
        # 模型的输出层。
        self.out = Sequential(Linear(64, 1))

    def forward(self, x):
        '''此函数采用输入张量，将其传递给递归神经网络 (RNN)，然后返回输出张量。
        
        Parameters
        ----------
        x
            前向方法的输入张量。它作为参数传递给该方法，预计是形状为 (sequence_length, batch_size, input_size) 的张量，其中 sequence_length
        是输入序列的长度，batch_size 是一个批次中的序列数，输入
        
        Returns
        -------
            处理输入序列 x 后 RNN 模型的输出。具体来说，它返回RNN模型最后一个时间步的输出，经过一个线性层（self.out）得到最终的输出。
        
        '''
        # `r_out, (h_n, h_c) = self.rnn(x, None)` 将输入张量 `x` 通过 LSTM 层 `self.rnn` 并返回两个输出：`r_out` 和一个元组
        # `(h_n, h_c)`。 `r_out` 是 LSTM 层在每个时间步的输出，而 `(h_n, h_c)` 是 LSTM 层的最终隐藏状态和单元状态。 `None`
        # 参数作为初始隐藏状态和单元状态传递，表示 LSTM 层应将它们初始化为零。
        r_out, (h_n, h_c) = self.rnn(x, None)
        # `out = self.out(r_out)` 将来自 LSTM 层的输出张量 `r_out` 通过线性层 `self.out` 传递，以获得 RNN
        # 模型的最终输出。线性层对输入张量应用线性变换并产生大小为 1 的输出张量，这是下一时间步长的预测值。
        out = self.out(r_out)
        return out


# TrainSet 类是一个 PyTorch 数据集，它接收数据并将数据和标签作为浮点数返回。
class TrainSet(Dataset):
    def __init__(self, data):
        '''此函数使用输入数据初始化对象的数据和标签属性，其中数据是二维张量，标签是一维张量。
        
        Parameters
        ----------
        data
            包含输入数据和标签的张量。输入数据是一个矩阵，其中每一行代表一个样本，每一列代表一个特征。矩阵的最后一列包含每个样本的标签。
        
        '''
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        '''此函数返回给定索引处的数据和标签值的元组。
        
        Parameters
        ----------
        index
            索引参数是我们要从数据和标签数组中检索的元素的位置。它在 __getitem__ 方法中用于访问该索引位置处的相应数据和标签值。
        
        Returns
        -------
            包含对象中指定索引处的数据和标签的元组。
        
        '''
        return self.data[index], self.label[index]

    def __len__(self):
        '''此函数返回对象数据属性的长度。
        
        Returns
        -------
            对象的“数据”属性的长度。
        
        '''
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
