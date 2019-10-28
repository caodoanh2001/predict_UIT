import os
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
# NHẬP DỮ LIỆU
def read_data(csv_file):
    csv_df = pd.read_csv(csv_file)
    return csv_df
data = read_data('diem_chuan.csv')
id = data.iloc[:, :1]
i = id.values
del data['ma_nganh']

def split_data(csv_df):
    input = csv_df.iloc[:, :-1] # lấy trên mọi hàng, và chỉ bỏ cột cuối cùng
    output = csv_df.iloc[:, -1] # lấy trên mọi hàng, và chỉ lấy cột cuối cùng
    x = input.values
    y = output.values.reshape((-1, 1)) # reshape để chuyển y thành ma trận cột
    return x, y
def find_optimize(input, outcome):
    """
    input.T: chuyển vị của ma trận input
    np.dot(a,b) : nhân từng phần tử của ma trận a với ma trận b
    np.linalg.pinv(x): tìm giả ngịch đảo/ ngịch đảo của ma trận x
    """
    w = np.dot(np.linalg.pinv(np.dot(input.T, input)), np.dot(input.T, outcome))
    return w
def optimize_with_sklearn(input, outcome):
    regr = linear_model.LinearRegression(fit_intercept=False)  # fit_intercept = False for calculating the bias
    regr.fit(input, outcome)
    return regr.coef_
    
data_array = split_data(data)
input = data_array[0]
outcome = data_array[1]
one = np.ones((input.shape[0], 1))
input = np.concatenate((one, input), axis=1)
w = find_optimize(input, outcome)
label = []
predict = []
cost = 0
y_hat = np.dot(input, w)
for i, x, y in zip(i, outcome, y_hat):
    label.append(x[0])
    predict.append(y[0])
    print(i,'Diem chuan 2019:', x[0], 'Diem du doan:', y[0])
    cost += pow(x[0] - y[0], 2)
print('Loss value',cost/2)
