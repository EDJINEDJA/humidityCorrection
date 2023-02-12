import torch
from src.trainer_ import model_pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = dict(
    epochs=1000,
    batch_size=128,
    learning_rate=0.005,
    output_length = 1,
    device = device , 
    withlogtrans = False , 
    method_scaling = "std")

if __name__ == "__main__":
    # Build, train and analyze the model with the pipeline
    model = model_pipeline(config)