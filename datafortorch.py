import numpy as np
import pandas as pd
import torch
# from torch.utils.data import Dataset,DataLoader
from torchtext.data import Dataset, Example

class TextDataset(Dataset):
    def __init__(self,root_dir,freq_min=3) :
        self.root_dir = root_dir
        self.freq_min =freq_min

    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        label = self.labels[idx]
        text = self.text[idx]
        sample = {"Text":text, "Senti": label}
        return sample
def get_loader():
    return
class

def main():
    text = ['Happy','Amazing','Sad']
    labels = ['positive','positive','negative'] 
    text_labels_df = pd.DataFrame({'Text':text,'Labels':labels})
    TD=TextDataset(text_labels_df['Text'],text_labels_df['Labels'])
    print(next(iter(TD)))
    print(len(TD))
    print(list(DataLoader(TD)))

if __name__ == "__main__":
    main()