import numpy as np
import onnx

def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tobytes(),
        raw=True)

    return initializer_tensor

def main() -> None:
    lm = np.fromfile('/home/yanxing/LLMExporter/onnx/onnx__MatMul_7', dtype=np.float32, count=-1, offset=0)
    lm = lm.reshape(4096, -1) # shape is (4096, 151936)
    # 4096 * 151936 * 4 is too big, split to 2 conv
    split_size = 151936 // 2
    w1 = lm[:, :split_size].transpose((1, 0)) # shape is (x, 4096)
    w2 = lm[:, split_size:].transpose((1, 0)) # shape is (x, 4096)
    ic = 4096
    # oc = lm.shape[0]
    model_input_name = "input"
    X = onnx.helper.make_tensor_value_info(model_input_name,
                                           onnx.TensorProto.FLOAT,
                                           [1, ic, 1, 1])
    model_output_name = "output"
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.INT64,
                                           [1, 1, 1, 1])
    conv1_output_node_name = "conv1"
    conv1_in_channels = ic
    conv1_out_channels = w1.shape[0]
    conv1_kernel_shape = (1, 1)
    conv1_pads = (0, 0, 0, 0)
    conv1_W = w1.reshape((conv1_out_channels, conv1_in_channels, *conv1_kernel_shape)).astype(np.float32)
    conv1_W_initializer_tensor_name = "conv1_w"
    conv1_W_initializer_tensor = create_initializer_tensor(
        name=conv1_W_initializer_tensor_name,
        tensor_array=conv1_W,
        data_type=onnx.TensorProto.FLOAT)
    conv1_node = onnx.helper.make_node(
        name="Conv1",
        op_type="Conv",
        inputs=[
            model_input_name, conv1_W_initializer_tensor_name,
        ],
        outputs=[conv1_output_node_name],
        kernel_shape=conv1_kernel_shape,
        pads=conv1_pads,
    )

    conv2_output_node_name = "conv2"
    conv2_in_channels = ic
    conv2_out_channels = w2.shape[0]
    conv2_kernel_shape = (1, 1)
    conv2_pads = (0, 0, 0, 0)
    conv2_W = w2.reshape((conv2_out_channels, conv2_in_channels, *conv2_kernel_shape)).astype(np.float32)
    conv2_W_initializer_tensor_name = "conv2_w"
    conv2_W_initializer_tensor = create_initializer_tensor(
        name=conv2_W_initializer_tensor_name,
        tensor_array=conv2_W,
        data_type=onnx.TensorProto.FLOAT)
    conv2_node = onnx.helper.make_node(
        name="Conv2",
        op_type="Conv",
        inputs=[
            model_input_name, conv2_W_initializer_tensor_name,
        ],
        outputs=[conv2_output_node_name],
        kernel_shape=conv2_kernel_shape,
        pads=conv2_pads,
    )

    concat_output_node_name = "concat_output"
    concat_node = onnx.helper.make_node(
        name="Concat",
        op_type="Concat",
        inputs=[conv1_output_node_name, conv2_output_node_name],
        outputs=[concat_output_node_name],
        axis=1,
    )

    argmax_output_node_name = "output"
    argmax_node = onnx.helper.make_node(
        name="ArgMax",
        op_type="ArgMax",
        inputs=[concat_output_node_name],
        outputs=[argmax_output_node_name],
        axis=1,
    )
    graph_def = onnx.helper.make_graph(
        nodes=[conv1_node, conv2_node, concat_node, argmax_node],
        name="lm",
        inputs=[X],
        outputs=[Y],
        initializer=[
            conv1_W_initializer_tensor, conv2_W_initializer_tensor
        ],
    )
    model_def = onnx.helper.make_model(graph_def, producer_name="mnn-lm")
    model_def.opset_import[0].version = 15
    # model_def = onnx.shape_inference.infer_shapes(model_def)
    onnx.save(model_def, "lm.onnx", save_as_external_data=True)
if __name__ == "__main__":
    main()