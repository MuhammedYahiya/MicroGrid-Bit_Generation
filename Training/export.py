import os
import numpy as np
import torch
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

def export_model(model_for_export, model_dir):
    ready_model_filename = os.path.join(model_dir, "model.onnx")

    input_shape = (1, 5)

    input_a = np.random.randint(0, 1, size=input_shape).astype(np.float32)
    input_a = 2 * input_a - 1
    scale = 1.0
    input_t = torch.from_numpy(input_a * scale)

    model_for_export.cpu()

    export_qonnx(
        model_for_export, export_path=ready_model_filename, input_t=input_t
    )

    qonnx_cleanup(ready_model_filename, out_file=ready_model_filename)
    model = ModelWrapper(ready_model_filename)

    model.set_tensor_datatype(model.graph.input[0].name, DataType["BIPOLAR"])
    model = model.transform(ConvertQONNXtoFINN())
    model.save(ready_model_filename)
    print("Model saved to %s" % ready_model_filename)