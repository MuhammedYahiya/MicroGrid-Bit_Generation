import numpy as np
import torch
from sklearn.metrics import accuracy_score

def train(model, train_loader, optimizer, criterion, device):
    losses = []
    model.train()

    for i, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(inputs.float())
        loss = criterion(output, target.unsqueeze(1))

        loss.backward()
        optimizer.step()

        losses.append(loss.data.cpu().numpy())
    return losses

def test(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs.float())
            output = torch.sigmoid(output)
            pred = (output.detach().cpu().numpy() > 0.5) * 1
            target = target.cpu().float()
            y_true.extend(target.tolist())
            y_pred.extend(pred.reshape(-1).tolist())

    return accuracy_score(y_true, y_pred)

def test_bipolar(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
   
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            
            input_scaled = 2 * inputs - 1
            
            output = model(input_scaled.float())
            
            output_bipolar = (output > 0.5).float() * 2 - 1
            y_pred.extend(output_bipolar.flatten().cpu().numpy())
            
            expected = 2 * target.float() - 1
            y_true.extend(expected.cpu().numpy())
        
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"Some True Labels: {y_true[:10]}")
    print(f"Some Predictions: {y_pred[:10]}")
    print(f"Accuracy: {accuracy}")
    
    return accuracy