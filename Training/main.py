import torch
from torch.utils.data import DataLoader

from data_preparation import get_preqnt_datasets
from model import create_model

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