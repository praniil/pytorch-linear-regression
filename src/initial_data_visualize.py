import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("../Dataset/train_data.csv")
train_features_x = train_data['X']
print(len(train_features_x))

train_label_y = train_data['y']
print(len(train_label_y))

test_data = pd.read_csv("../Dataset/test_data.csv")
test_features_x = test_data['X']
test_label_y = test_data['y']


def plot_predictions(train_data=train_features_x, 
                     train_labels=train_label_y, 
                     test_data=test_features_x, 
                     test_labels=test_label_y, 
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  print("in plot predictions")
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=2, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14})
  plt.savefig("../Results/initial_data_visualization.png")



if __name__ == "__main__":
    plot_predictions()
