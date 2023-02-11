from input.preprocessing import load_transform
from configparser import ConfigParser 
from src.model.Network import Network
from src.trainer import trainer
import torch

batch_size = 128
n_epochs = 1000
learning_rate = 0.0001
const_variables = ConfigParser()
const_variables.read("./config/config.ini")
output_length = int(const_variables.get("slicing","output_length"))


if __name__ == "__main__":
    parser = load_transform()
    scaled_array , X_train,y_train,X_test,y_test = parser.fit_transform(withlogtrans=False , method_scaling="minmax")
   
    n_features = scaled_array.shape[-1]
    
    model= Network(n_features=int(n_features),
                                output_length=output_length,
                                batch_size = batch_size)

    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
   
    trainer(X_train,y_train,n_epochs,batch_size, model ,criterion, optimizer)