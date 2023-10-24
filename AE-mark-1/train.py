import torch
import numpy as np
import hydra
from torch.utils.data import DataLoader

from src.auto import AE
from utils.data import ImageDataset



@hydra.main(version_base=None,config_path="config",config_name="config")
def train (cfg):
    dataset=ImageDataset(cfg.data.root_dir,cfg.params.csv_dir)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [cfg.data.train_split,cfg.data.test_split])


    model = AE().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.params.LR)
    train_loader = DataLoader(train_dataset, shuffle=cfg.data.train_shuffle, batch_size=cfg.data.batch_size)
    test_loader = DataLoader(test_dataset, shuffle=cfg.data.test_shuffle, batch_size=cfg.data.batch_size)
    for epoch in range(cfg.params.no_epochs):
        mean_epoch_loss=[]
        for batch in train_loader:



            
    
            batch_image , batch_depth = batch
            batch_depth=batch_depth.to("cuda")
            batch_image=batch_image.to("cuda")
  

            predicted_depth = model(batch_image)
    
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(batch_depth, predicted_depth) 
            mean_epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
    
        if epoch % cfg.params.print_fre == 0:
            print('---')
            print(f"Epoch: {epoch} | Train Loss {np.mean(mean_epoch_loss)}")
            torch.save(model,"out/model.pt")





if __name__=="__main__":
    train()

































#image2=Image.open("image2.jpg")
#image1=Image.open("image1.jpg")


#transform = transforms.Compose([
    #transforms.ToTensor(), # Convert to torch tensor (scales data into [0,1])
    #transforms.Lambda(lambda t: (t * 2) - 1), # Scale data between [-1, 1] 
#])


#reverse_transform = transforms.Compose([
    #transforms.Lambda(lambda t: (t + 1) / 2), # Scale data between [0,1]
    #transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
   # transforms.Lambda(lambda t: t * 255.), # Scale data between [0.,255.]
  #  transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)), # Convert into an uint8 numpy array
 #   transforms.ToPILImage(), # Convert to PIL image
#])

#torch_image2 = transform(image2)
#torch_image1 = transform(image1)






