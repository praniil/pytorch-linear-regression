from build_model import LinearRegressionModel
import torch
import pandas as pd

# instantiate a fresh instance of LinearRegressionModel
loaded_model = LinearRegressionModel()

loaded_model.load_state_dict(torch.load("../trained_model/trained_linear_regression.pth"))
device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_model.to(device)

train_data = pd.read_csv("../Dataset/train_data.csv") 
test_data = pd.read_csv("../Dataset/test_data.csv")

X_test = torch.tensor(test_data['X'].values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(test_data['y'].values, dtype=torch.float32).unsqueeze(1)

X_test = X_test.to(device)
y_test = y_test.to(device)


#evaluation mode
loaded_model.eval()
with torch.inference_mode():
    loaded_model_pred = loaded_model(X_test)
y_preds = loaded_model_pred
print(y_preds)