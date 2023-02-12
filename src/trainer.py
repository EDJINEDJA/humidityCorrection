import torch

def trainer(X_train,y_train,X_test,y_test,n_epochs,batch_size,model ,criterion, optimizer):
    print("______________training part _________________")
    model.train()
    for e in range(1, n_epochs+1):
        for b in range(0, len(X_train), batch_size):
            features = X_train[b:b+batch_size,:,:]
            target = y_train[b:b+batch_size]    

            X_batch = torch.tensor(features,dtype=torch.float32)    
            y_batch = torch.tensor(target,dtype=torch.float32)

            output = model(X_batch)

            loss = criterion(output.view(-1), y_batch.view(-1))  

            loss.backward()
            optimizer.step()        
            optimizer.zero_grad() 

        if e % 10== 0:
           print('epoch :: training', e, 'loss: ', loss.item())
    
    print("______________evaluation part _________________")
    model.eval()
    for e in range(1, n_epochs+1):
        for b in range(0, len(X_test), batch_size):
            features = X_test[b:b+batch_size,:,:]
            target = y_test[b:b+batch_size]    

            X_batch = torch.tensor(features,dtype=torch.float32)    
            y_batch = torch.tensor(target,dtype=torch.float32)

            output = model(X_batch)
            
            loss = criterion(output.view(-1), y_batch.view(-1))  

        if e % 10== 0:
            print('epoch :: eval', e, 'loss: ', loss.item())