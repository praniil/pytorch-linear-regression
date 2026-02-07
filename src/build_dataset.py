import matplotlib.pyplot as plt
import torch
import csv
import pandas as pd
    
# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start  = 0
end = 1
step = 0.02
X = torch.arange(start, end, step)
X = torch.unsqueeze(X, dim= 1)
y =  weight * X + bias

print(X[:10])
print(y[:10])
print(X.size())
print(y.size())

#split the data into the train and the test set
train_split = int(0.8 * len(X))
print(train_split)
X_train = X[:train_split]
y_train = y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test)) 

#writing in csv file
train_data = {
        'X': X_train.squeeze().numpy(),
        'y': y_train.squeeze().numpy()
}

df = pd.DataFrame(train_data)
df.to_csv('../Dataset/train_data.csv', index=[0, len(X)])

print(df)


