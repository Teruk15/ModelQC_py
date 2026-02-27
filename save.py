from cnn import CNN
import os
import sys
import torch

def main():
    # Load checkpoint
    checkpointPath = './model/checkpoint.pth'
    savePath = './model'
    
    if not os.path.exists(checkpointPath):
        print(f'{checkpointPath} does not exist')
        sys.exit(1)
        
    ckpt = torch.load(checkpointPath, map_location="cpu")

    model = CNN(num_classes=ckpt["num_classes"])   # must match your architecture
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    N = 1
    C = 1
    H = 3
    W = 4800

    # make a dummy input with the SAME shape you use in inference
    dummy_input = torch.zeros(N, C, H, W)

    save_file = os.path.join(savePath, "model.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        save_file,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,
    )


if __name__ == "__main__":
    main()