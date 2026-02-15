from build_model import LinearRegressionModel
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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

# Calculate test loss
loss_function = torch.nn.L1Loss()
test_loss = loss_function(y_preds, y_test)

# Print results
print(f"Test Loss (MAE): {test_loss.item():.4f}")
print(f"\nSample Predictions (first 10):")
print(y_preds[:10])

# Save predictions to CSV
results_df = pd.DataFrame({
    'X_test': test_data['X'].values,
    'y_actual': test_data['y'].values,
    'y_predicted': y_preds.cpu().numpy().flatten()
})
results_df.to_csv("../Results/test_predictions.csv", index=False)
print(f"\nPredictions saved to ../Results/test_predictions.csv")

# Create visualization
plt.figure(figsize=(10, 7))
plt.scatter(train_data['X'], train_data['y'], c="b", s=2, label="Training data", alpha=0.4)
plt.scatter(test_data['X'], test_data['y'], c="g", s=4, label="Testing data (actual)")
plt.scatter(test_data['X'], y_preds.cpu().numpy(), c="r", s=4, label="Predictions")
plt.xlabel("X")
plt.ylabel("y")
plt.title(f"Linear Regression Results\nTest Loss (MAE): {test_loss.item():.4f}")
plt.legend(prop={"size": 12})
plt.savefig("../Results/test_results_visualization.png", dpi=300, bbox_inches='tight')
print(f"Visualization saved to ../Results/test_results_visualization.png")

# Save test metrics
metrics = {
    'Test Loss (MAE)': [test_loss.item()],
    'Number of Test Samples': [len(test_data)],
    'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("../Results/test_metrics.csv", index=False)
print(f"Test metrics saved to ../Results/test_metrics.csv")