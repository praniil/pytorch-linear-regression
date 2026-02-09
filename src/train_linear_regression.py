import torch
import pandas as pd
import torch.nn as nn
from build_model import LinearRegressionModel, init_model
from pprint import pprint

torch.manual_seed(42)

# number of epochs
epochs = 1000

train_data = pd.read_csv("../Dataset/train_data.csv") 
test_data = pd.read_csv("../Dataset/test_data.csv")

X_train = train_data['X']
y_train = train_data['y']
X_test = test_data['X']
y_test = test_data['y']

X_train = torch.tensor(train_data['X'].values, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(train_data['y'].values, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(test_data['X'].values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(test_data['y'].values, dtype=torch.float32).unsqueeze(1)


model = LinearRegressionModel()
parameters = model.parameters()

# loss function
loss_function = nn.L1Loss()

# optimizer
optimizer = torch.optim.SGD(params=parameters, lr=0.005)
for epoch in range(epochs):
    # Training
    model.train()

    #forward passs
    y_pred = model(X_train)
    # loss calculation
    train_loss = loss_function(y_pred, y_train)

    #clears the prev gradiant
    optimizer.zero_grad()

    # writes new value of grads
    train_loss.backward()

    # update the weights
    optimizer.step()

    # Testing
    model.eval()

    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_function(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {train_loss} | Test loss: {test_loss}")

torch.save(obj=model.state_dict(), f="../trained_model/trained_linear_regression.pth")