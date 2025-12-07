import torch
from ppo import ActorCriticNetwork

MODEL_PATH = "soccer_final.pth" # the model path for the trained AI
ONNX_OUTPUT = "web_demo/soccer_ai.onnx" # output ONNX model path

STATE_DIM = 22 + 1 # original state dim + 1 for player side indicator
ACTION_DIM = 9 # number of discrete actions

# load and move model to CPU
device = torch.device("cpu")
model = ActorCriticNetwork(state_dim=STATE_DIM, action_dim=ACTION_DIM)
model.to(device)

# load the trained model and set to eval mode
checkpoint = torch.load(MODEL_PATH, map_location=device)
if 'agent_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['agent_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# create a dummy input for the model
dummy_input = torch.randn(1, STATE_DIM, device=device)

# export the model to ONNX format
torch.onnx.export(
    model.actor, # only export the actor part for inference                   
    dummy_input, # the input to the model
    ONNX_OUTPUT, # where to save the ONNX model
    export_params=True, # store the trained parameter weights inside the model file
    opset_version=17, # the ONNX version to export the model to
    do_constant_folding=True, # whether to execute constant folding for optimization
    input_names=["state"], # the input tensor name
    output_names=["logits"], # the output tensor name
    dynamic_axes={"state": {0: "batch"}, "logits": {0: "batch"}} # variable length axes
)

print(f"Model converted and saved to {ONNX_OUTPUT}")