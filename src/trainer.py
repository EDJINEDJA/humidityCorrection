import torch

def trainer(X_train,y_train,n_epochs,batch_size,model ,criterion, optimizer):
    model.train()
    for e in range(1, n_epochs+1):
        for b in range(0, len(X_train), batch_size):
            features = X_train[b:b+batch_size,:,:]
            target = y_train[b:b+batch_size]    

            X_batch = torch.tensor(features,dtype=torch.float32)    
            y_batch = torch.tensor(target,dtype=torch.float32)

            output = model(X_batch)
            # print(output.shape)
            # print(y_batch.shape)
            loss = criterion(output.view(-1), y_batch.view(-1))  

            loss.backward()
            optimizer.step()        
            optimizer.zero_grad() 

        if e % 2== 0:
           print('epoch', e, 'loss: ', loss.item())