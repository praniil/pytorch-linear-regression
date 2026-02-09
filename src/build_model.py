import torch.nn as nn
import torch    
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

def init_model():
    #setting the manual seed
    torch.manual_seed(42)
    model = LinearRegressionModel()
    print(model.weights)
    print(model)
    print(model.state_dict())


    # check the model device
    print(next(model.parameters()).device)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # changing it to the gpu
    model.to(device=device)
    print(next(model.parameters()).device)
    
    return model

if __name__ == "__main__":
    init_model()