import wandb
import torch
from input.preprocessing import load_transform
from src.model.Network import Network
from tqdm.auto import tqdm
import os

from sklearn.metrics import r2_score

def make(config):
    # Make the data
    parser = load_transform()
    X_train_loader ,y_train_loader ,X_test_loader ,y_test_loader = parser.fit_transform(config.withlogtrans,config.method_scaling)
    
    # Make the model
    n_features = X_train_loader.shape[-1]
    model= Network(n_features=int(n_features),
                                output_length=config.output_length,
                                batch_size = config.batch_size)

    # Make the loss and optimizer
    #criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, X_train_loader ,y_train_loader ,X_test_loader ,y_test_loader, criterion, optimizer


def train_test_log(loss, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss})
  

def trainer(model, X_train ,y_train ,X_test, y_test, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    model.train()
    for e in tqdm(range(1, config.epochs+1)):
        for b in range(0, len(X_train), config.batch_size):
            features = X_train[b:b+config.batch_size,:,:]
            target = y_train[b:b+config.batch_size]    

            X_batch = torch.tensor(features,dtype=torch.float32)
            y_batch = torch.tensor(target,dtype=torch.float32)  

            output = model(X_batch)

            loss = criterion(output.view(-1), y_batch.view(-1))  

            loss.backward()
            optimizer.step()        
            optimizer.zero_grad() 

        if e % 10== 0:
            train_test_log(loss.item(),e)

       

    model.eval()
    # Run the model on some test examples
    with torch.no_grad():
        
        for e in tqdm(range(1, config.epochs+1)):
            for b in range(0, len(X_test), config.batch_size):
                features = X_test[b:b+config.batch_size,:,:]
                target = y_test[b:b+config.batch_size]    

                X_batch = torch.tensor(features,dtype=torch.float32) 
                y_batch = torch.tensor(target,dtype=torch.float32)

                output = model(X_batch)
                
                loss = criterion(output.view(-1), y_batch.view(-1))  

            if e % 10== 0:
               train_test_log(loss.item(),e)

      

    # Save the model in the exchangeable ONNX format
    torch.save(model.state_dict(), os.path.abspath("src/output/model.pth"))
    

def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="HumidityCorrectionProject", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, X_train_loader ,y_train_loader ,X_test_loader ,y_test_loader, criterion, optimizer = make(config)
      

      # and use them to train the model
      trainer(model, X_train_loader ,y_train_loader  ,X_test_loader ,y_test_loader,criterion, optimizer, config)

    return model