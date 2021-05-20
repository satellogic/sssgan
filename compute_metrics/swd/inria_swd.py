from swd import swd
import torch
from PIL import Image
import argparse
import glob as glob
import os 
import torchvision.transforms as transforms
from tqdm import tqdm

def image_collection_to_torch(path):
    imgs_path = glob.glob(os.path.join(path, "*.jpg"))
    if len(imgs_path) == 0:
        imgs_path = glob.glob(os.path.join(path, "*.png"))

    acum_tensor = []
    i=0
    for img_path in tqdm(imgs_path):
        i += 1
        img = Image.open(img_path)
        transform = transforms.ToTensor()
        t = transform(img)
        t = t.unsqueeze(0)
        acum_tensor.append(t)
        
    acum_tensor = torch.cat(acum_tensor,0)
    return acum_tensor

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--o", "--original", type=str, required=True)
    parser.add_argument("--f", "--fake", type=str, required=True)

    args = parser.parse_args()
    print("Prepare")
    original = image_collection_to_torch(args.o)
    fake = image_collection_to_torch(args.f)
    print("SWD")
    min_images = min(original.shape[0], fake.shape[0])
    original = original[:min_images]
    fake =fake[:min_images]
    print(original.shape)
    print(fake.shape)
    out = sswd(original, fake, device="cuda:1", return_by_resolution=True, pyramid_batchsize=128) # Fast estimation if device="cuda"
    print(out) 
