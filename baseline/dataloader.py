import numpy as np
import pandas as pd
import torch.utils.data as data
import os
import csv
from torchvision import transforms as T
from PIL import Image
from sklearn.model_selection import KFold
from opts import get_opts

# save img path to one csv file
def save2csv(path, csvname='anime_data.csv'):
    # read imgname in dirt
    path_list = os.listdir(path)
    path_list.sort(key=lambda x:int(x.split('frame_')[1].split('.jpeg')[0]))
    # save img to csv
    if os.path.exists(csvname):
        os.remove(csvname) # delete file if exists
    f = open(csvname, 'a', newline='')
    for filename in path_list:
        writer = csv.writer(f)
        writer.writerow([filename])
    print("CSV file has been created!")
    f.close()

# Image normalization and data augmentation
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
transform_init = T.Compose([T.ToTensor(), normalize, T.Resize((128, 128))])

# Define Dataset
# Maybe we can use ImageFolder and it will be faster than using Image to read imgs every time
class Dataload(data.Dataset):
    def __init__(self, csv_name, imgpath, transform=transform_init):
        super(Dataload, self).__init__()
        self.csv_name = csv_name
        self.transform = transform
        self.path = imgpath + np.array(pd.read_csv(csv_name, header=None))

    def __getitem__(self, index):

        img = Image.open(self.path[index][0])
        img = self.transform(img)
        return img.type('torch.FloatTensor')

    def __len__(self):
        return self.path.shape[0]


