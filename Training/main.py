
from data_preparation import get_preqnt_datasets

if __name__ == "__main__":
    csv_path = '/home/admin_eee/finn/microgrid/V2G_G2V.csv'
    response_column = 'response'
    train_quantized_dataset, test_quantized_dataset = get_preqnt_datasets(csv_path, response_column)
    
    print("Samples in each set: train = %d, test = %d" % (len(train_quantized_dataset), len(test_quantized_dataset)))
    print("Shape of one input sample: " + str(train_quantized_dataset[0][0].shape))
