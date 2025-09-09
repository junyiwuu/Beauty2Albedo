import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


'''
normalized image should between -1 and 1
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
'''

class BeautyAlbedoDataset(Dataset):
    def __init__(self, dataset_dir: str, filename_ls_path:str 
                 ):
        
        super().__init__()

        self.dataset_dir = dataset_dir
        self.filename_ls_path = filename_ls_path

        with open(self.filename_ls_path, "r") as f:
            lines = [line.strip().split() for line in f if line.strip()]
        
        self.beauty_files = [os.path.join(dataset_dir, b) for b, _ in lines]
        self.albedo_files = [os.path.join(dataset_dir, a) for _ , a in lines]
        
        assert len(self.beauty_files) !=0 , "no found beauty files"
        assert len(self.albedo_files) !=0 , "no found albedo files"

        self.to_tensor = transforms.ToTensor()
        self.disp_name = "Beauty_Albedo"
        
        
    def __len__(self):
        return len(self.beauty_files)

    def __getitem__(self, idx):
       
        beauty_path = self.beauty_files[idx]
        albedo_path = self.albedo_files[idx]

        name = os.path.basename(beauty_path)

        beauty_img = Image.open(beauty_path).convert("RGB")
        albedo_img = Image.open(albedo_path).convert("RGB")

        albedo_tensor = self.to_tensor(albedo_img) * 2.0 - 1.0
        beauty_tensor = self.to_tensor(beauty_img) * 2.0 - 1.0

        # the original int verison of beauty , albedo for validation
        # beauty_int = torch.from_numpy(np.array(beauty_img)).permute(2, 0, 1).int()
        beauty_int = beauty_tensor
        

        # all 1 mask 
        valid_mask = torch.ones_like(beauty_tensor[0:1, :, :],  dtype=torch.bool )

        return {
            "beauty_norm": beauty_tensor, 
            "albedo_norm": albedo_tensor, 
            "valid_mask_raw": valid_mask,
            "rgb_relative_path" : name,
            "beauty_int": beauty_int,
            
            }

