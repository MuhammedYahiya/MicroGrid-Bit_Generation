import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
import numpy as np
from copy import deepcopy

from data_preparation import get_preqnt_datasets
from model import create_model,CybSecMLPForExport
from train import train, test, test_bipolar
from export import export_model

if __name__ == "__main__":
    csv_path = '/home/admin_eee/finn/microgrid/V2G_G2V.csv'
    response_column = 'response'
    train_quantized_dataset, test_quantized_dataset = get_preqnt_datasets(csv_path, response_column)
    
    print("Samples in each set: train = %d, test = %d" % (len(train_quantized_dataset), len(test_quantized_dataset)))
    print("Shape of one input sample: " + str(train_quantized_dataset[0][0].shape))

    batch_size = 32
    
    train_quantized_loader = DataLoader(train_quantized_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_quantized_loader = DataLoader(test_quantized_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


    for x, y in train_quantized_loader:
        print("Input shape for 1 batch: " + str(x.shape))
        print("Label shape for 1 batch: " + str(y.shape))
        break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Target device: " + str(device))
    
    input_size = 5
    hidden1 = 64
    hidden2 = 64
    weight_bit_width = 4
    act_bit_width = 4
    num_classes = 1

    model = create_model(input_size, hidden1, hidden2, weight_bit_width, act_bit_width, num_classes)
    model.to(device)
    
    num_epochs = 10
    lr = 0.001

    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    
    running_loss = []
    running_test_acc = []
    t = trange(num_epochs, desc="Training loss", leave=True)
    
    for epoch in t:
        loss_epoch = train(model, train_quantized_loader, optimizer, criterion, device)
        test_acc = test(model, test_quantized_loader, device)
        scheduler.step(np.mean(loss_epoch))
        t.set_description("Training loss = %f test accuracy = %f" % (np.mean(loss_epoch), test_acc))
        t.refresh()
        running_loss.append(loss_epoch)
        running_test_acc.append(test_acc)

    torch.save(model.state_dict(), "state_dict_self-trained.pth")
    
    model = model.cpu()
    modified_model = deepcopy(model)

    W_orig = modified_model[0].weight.data.detach().numpy()
    print(f"Original Weight Shape: {W_orig.shape}")

    W_new = W_orig
    print(f"New Weight Shape: {W_new.shape}")

    modified_model[0].weight.data = torch.from_numpy(W_new)

    model_for_export = CybSecMLPForExport(modified_model)
    model_for_export.to(device)

    binary_quantization_accuracy = test_bipolar(model_for_export, test_quantized_loader, device)

    print("Binary Quantization Accuracy: ", binary_quantization_accuracy)

    model_dir = "/home/admin_eee/finn/microgrid/model"
    export_model(model_for_export, model_dir)