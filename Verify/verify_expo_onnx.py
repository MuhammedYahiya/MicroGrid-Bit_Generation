
import torch
import os
import pandas as pd
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from torch.utils.data import TensorDataset
from brevitas.nn import QuantLinear, QuantReLU
import torch.nn as nn
import finn.core.onnx_exec as oxe
from tqdm import trange

model_dir = "/home/admin_eee/finn/microgrid/model"
ready_model_filename = os.path.join(model_dir,"model.onnx")
model_for_sim = ModelWrapper(ready_model_filename)

dir(model_for_sim)

finnonnx_in_tensor_name = model_for_sim.graph.input[0].name
finnonnx_out_tensor_name = model_for_sim.graph.output[0].name
print("Input tensor name: %s" % finnonnx_in_tensor_name)
print("Output tensor name: %s" % finnonnx_out_tensor_name)
finnonnx_model_in_shape = model_for_sim.get_tensor_shape(finnonnx_in_tensor_name)
finnonnx_model_out_shape = model_for_sim.get_tensor_shape(finnonnx_out_tensor_name)
print("Input tensor shape: %s" % str(finnonnx_model_in_shape))
print("Output tensor shape: %s" % str(finnonnx_model_out_shape))
finnonnx_model_in_dt = model_for_sim.get_tensor_datatype(finnonnx_in_tensor_name)
finnonnx_model_out_dt = model_for_sim.get_tensor_datatype(finnonnx_out_tensor_name)
print("Input tensor datatype: %s" % str(finnonnx_model_in_dt.name))
print("Output tensor datatype: %s" % str(finnonnx_model_out_dt.name))
print("List of node operator types in the graph: ")
print([x.op_type for x in model_for_sim.graph.node])

model_for_sim = model_for_sim.transform(InferShapes())
model_for_sim = model_for_sim.transform(FoldConstants())
model_for_sim = model_for_sim.transform(GiveUniqueNodeNames())
model_for_sim = model_for_sim.transform(GiveReadableTensorNames())
model_for_sim = model_for_sim.transform(InferDataTypes())
model_for_sim = model_for_sim.transform(RemoveStaticGraphInputs())

verif_model_filename = os.path.join(model_dir, "model_verification.onnx")
model_for_sim.save(verif_model_filename)

def get_preqnt_dataset(data_file: str):
    df = pd.read_csv(data_file).astype(np.float32)
    part_data_in = torch.from_numpy(df.iloc[:, :-1].values)  # All columns except the last one as inputs
    part_data_out = torch.from_numpy(df.iloc[:, -1].values)  # Last column as output
    return TensorDataset(part_data_in, part_data_out)

    
n_verification_inputs = 5
csv_data = "/home/admin_eee/finn/microgrid/V2G_G2V.csv"
test_quantized_dataset = get_preqnt_dataset(csv_data)
input_tensor = test_quantized_dataset.tensors[0][:n_verification_inputs]
input_tensor.shape

input_size = 5
hidden1 = 64
hidden2 = 64
weight_bit_width = 4
act_bit_width = 4
num_classes = 1


brevitas_model = nn.Sequential(
    nn.BatchNorm1d(input_size),  # Add input normalization
    QuantLinear(input_size, hidden1, bias=True, weight_bit_width=weight_bit_width),
    nn.BatchNorm1d(hidden1),
    nn.Dropout(0.3),
    QuantReLU(bit_width=act_bit_width),

    QuantLinear(hidden1, hidden2, bias=True, weight_bit_width=weight_bit_width),
    nn.BatchNorm1d(hidden2),
    nn.Dropout(0.3),
    QuantReLU(bit_width=act_bit_width),

    QuantLinear(hidden2, num_classes, bias=True, weight_bit_width=weight_bit_width)
)

trained_state_dict = torch.load("/home/admin_eee/finn/microgrid/Training/state_dict_self-trained.pth")
brevitas_model.load_state_dict(trained_state_dict, strict=False)

def inference_with_brevitas(current_inp):
    brevitas_output = brevitas_model.forward(current_inp)
    # apply sigmoid + threshold
    brevitas_output = torch.sigmoid(brevitas_output)
    brevitas_output = (brevitas_output.detach().numpy() > 0.5) * 1
    # convert output to bipolar
    brevitas_output = 2*brevitas_output - 1
    return brevitas_output

def inference_with_finn_onnx(current_inp):
    finnonnx_in_tensor_name = model_for_sim.graph.input[0].name
    finnonnx_model_in_shape = model_for_sim.get_tensor_shape(finnonnx_in_tensor_name)
    finnonnx_out_tensor_name = model_for_sim.graph.output[0].name
    
    # Convert input to numpy for FINN
    current_inp = current_inp.detach().numpy()
    
    # Quantize to BIPOLAR (-1 or 1)
    current_inp = np.sign(current_inp)
    
    # Ensure the input shape is correct (1,5)
    current_inp = current_inp.reshape(finnonnx_model_in_shape)
    
    # Create the input dictionary
    input_dict = {finnonnx_in_tensor_name: current_inp}
    
    # Run with FINN's execute_onnx
    output_dict = oxe.execute_onnx(model_for_sim, input_dict)
    
    # Get the output tensor
    finn_output = output_dict[finnonnx_out_tensor_name]
    
    return finn_output

# Helper function to print tensor info
def print_tensor_info(tensor, name):
    print(f"{name} shape: {tensor.shape}")
    print(f"{name} dtype: {tensor.dtype}")
    print(f"{name} min: {tensor.min()}, max: {tensor.max()}")
    print(f"{name} unique values: {np.unique(tensor)}")
    print(f"{name} first few values: {tensor.flatten()[:10]}")
    print()

verify_range = trange(n_verification_inputs, desc="FINN execution", position=0, leave=True)
brevitas_model.eval()

ok = 0
nok = 0

for i in verify_range:
    # Run in Brevitas with PyTorch tensor
    current_inp = input_tensor[i].reshape((1, 5))
    
    print(f"Verification step {i+1}:")
    print_tensor_info(current_inp.numpy(), "Input")
    
    brevitas_output = inference_with_brevitas(current_inp)
    print_tensor_info(brevitas_output, "Brevitas output")
    
    finn_output = inference_with_finn_onnx(current_inp)
    print_tensor_info(finn_output, "FINN output")
    
    # Compare the outputs
    if np.array_equal(finn_output, brevitas_output):
        ok += 1
    else:
        nok += 1
    
    verify_range.set_description(f"ok {ok} nok {nok}")
    verify_range.refresh()
    
    print("-" * 50)

print(f"\nVerification complete: {ok} matches, {nok} mismatches")
if ok == n_verification_inputs:
    print("Verification succeeded. Brevitas and FINN-ONNX execution outputs are identical")
else:
    print("Verification failed. Brevitas and FINN-ONNX execution outputs are NOT identical")