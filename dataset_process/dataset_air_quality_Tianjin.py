import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import torchcde

def get_randmask(observed_mask, min_miss_ratio=0., max_miss_ratio=1.):
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    rand_for_mask = rand_for_mask.reshape(-1)
    sample_ratio = np.random.rand()
    sample_ratio = sample_ratio * (max_miss_ratio-min_miss_ratio) + min_miss_ratio
    num_observed = observed_mask.sum().item()
    num_masked = round(num_observed * sample_ratio)
    rand_for_mask[rand_for_mask.topk(num_masked).indices] = -1

    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return cond_mask

def get_hist_mask(observed_mask, for_pattern_mask=None, target_strategy='hybrid'):
    if for_pattern_mask is None:
        for_pattern_mask = observed_mask
    if target_strategy == "hybrid":
        rand_mask = get_randmask(observed_mask)

    cond_mask = observed_mask.clone()
    mask_choice = np.random.rand()
    if target_strategy == "hybrid" and mask_choice > 0.5:
        cond_mask = rand_mask
    else:  
        cond_mask = cond_mask * for_pattern_mask
    return cond_mask

Tianjin_attributes = [
    '1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008',
    '1009', '1010', '1011', '1012', '1013', '1014', '1015', '1016',
    '1017', '1018', '1019', '1020', '1021', '1022', '1023', '1024',
    '1025', '1026', '1027'
]

class AirQuality_Dataset(Dataset):
    def __init__(self, eval_length=36, target_dim=27, mode="train", target_strategy="random", validindex=0, missing_ratio=0.1, seed=1):
        self.eval_length = eval_length
        self.target_dim = target_dim
        self.mode = mode
        self.target_strategy = target_strategy

        np.random.seed(seed)  

        if mode == "train":
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
            flag_for_histmask = [0, 1, 0, 1, 0, 1, 0, 1] 
            month_list.pop(validindex)
            flag_for_histmask.pop(validindex) 
        elif mode == "valid":
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
            month_list = month_list[validindex : validindex + 1]
        elif mode == "test":
            month_list = [3, 6, 9, 12]
        self.month_list = month_list

        # create data for batch
        self.observed_data = []  # values (separated into each month)
        self.observed_mask = []  # masks (separated into each month)
        self.gt_mask = []  # ground-truth masks (separated into each month)
        self.index_month = []  # indicate month
        self.position_in_month = []  # indicate the start position in month (length is the same as index_month)
        self.valid_for_histmask = []  # whether the sample is used for histmask
        self.use_index = []  # to separate train/valid/test
        self.cut_length = []  # excluded from evaluation targets
        
        data = pd.read_csv("./data/air_quality/tianjin.csv",index_col="time",parse_dates=True) #8760*27
        data.index = pd.to_datetime(data.index)
        selected_data = data[Tianjin_attributes]
        ob_mask = ~np.isnan(selected_data.values)
        print("shape of ob_mask:", ob_mask.shape)
        print("ob_mask True:", np.sum(ob_mask))
        test_month = [3, 6, 9, 12]
        for j in test_month:
            selected_data = selected_data[selected_data.index.month != j]
        tmp_values = selected_data.values.reshape(-1, 27)
        tmp_masks = (~np.isnan(tmp_values)).reshape(-1, 27)

        self.mean = np.zeros(27)
        self.std = np.zeros(27)
        for k in range(27):
            tmp_data = tmp_values[:, k][tmp_masks[:, k] == 1] 
            self.mean[k] = tmp_data.mean()
            self.std[k] = tmp_data.std()

        path = "./data/air_quality/tianjin_air_quality_meanstd.pk"
        with open(path, "wb") as f:
            pickle.dump([self.mean, self.std], f)

        df = pd.read_csv("./data/air_quality/tianjin.csv",index_col="time",parse_dates=True)         
        for i in range(len(month_list)):
            current_df = df[df.index.month == month_list[i]] 
            print(f"Length of data for month {month_list[i]}: {len(current_df)}") 

            current_length = len(current_df) - eval_length + 1
            print(f"Current length for month {month_list[i]}: {current_length}")

            last_index = len(self.index_month)
            self.index_month += np.array([i] * current_length).tolist() 
            self.position_in_month += np.arange(current_length).tolist()

            if mode == "train":
                self.valid_for_histmask += np.array(
                    [flag_for_histmask[i]] * current_length
                ).tolist()

            c_mask = 1 - current_df.isnull().values 
            c_data = (
                (current_df.fillna(0).values - self.mean) / self.std
            ) * c_mask

            masks = c_mask.reshape(-1).copy()
            print(f"masks shape for month {masks.shape}:")
            obs_indices = np.where(masks)[0].tolist() 
            miss_indices = np.random.choice(
                obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
            )
            masks[miss_indices] = False 
            c_gt_mask = masks.reshape(c_mask.shape)

            self.observed_mask.append(c_mask)
            self.gt_mask.append(c_gt_mask)
            self.observed_data.append(c_data)
            print(f"Dimensions of observed_values: {[item.shape for item in self.observed_data]}")
            print(f"Dimensions of observed_masks: {[item.shape for item in self.observed_mask]}")
            print(f"Dimensions of gt_masks: {[item.shape for item in self.gt_mask]}")

            if mode == "test":
                n_sample = len(current_df) // eval_length

                c_index = np.arange(
                    last_index, last_index + eval_length * n_sample, eval_length
                )

                self.use_index += c_index.tolist()
                self.cut_length += [0] * len(c_index)
                if len(current_df) % eval_length != 0:  # avoid double-count for the last time-series
                    self.use_index += [len(self.index_month) - 1]
                    self.cut_length += [eval_length - len(current_df) % eval_length]

        if mode != "test":
            self.use_index = np.arange(len(self.index_month))
            self.cut_length = [0] * len(self.use_index)

        
        if mode == "train":
            ind = -1
            self.index_month_histmask = []
            self.position_in_month_histmask = []

            for i in range(len(self.index_month)): # 4842
                while True:
                    ind += 1
                    if ind == len(self.index_month):
                        ind = 0
                    if self.valid_for_histmask[ind] == 1: 
                        self.index_month_histmask.append(self.index_month[ind]) 
                                                                                
                        self.position_in_month_histmask.append(
                            self.position_in_month[ind] 
                        )
                        break
        else:  
            self.index_month_histmask = self.index_month
            self.position_in_month_histmask = self.position_in_month


    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        c_month = self.index_month[index]
        c_index = self.position_in_month[index]
        hist_month = self.index_month_histmask[index]
        hist_index = self.position_in_month_histmask[index]

        ob_data = self.observed_data[c_month][c_index:c_index + self.eval_length]
        ob_mask = self.observed_mask[c_month][c_index : c_index + self.eval_length]
        ob_mask_t = torch.tensor(ob_mask).float()
        gt_mask = self.gt_mask[c_month][c_index : c_index + self.eval_length]

        for_pattern_mask = self.observed_mask[hist_month][hist_index : hist_index + self.eval_length]

        if self.mode != 'train':
            cond_mask = torch.tensor(gt_mask).to(torch.float32)
        else:
            if self.target_strategy != 'random':
                cond_mask = get_hist_mask(ob_mask_t, for_pattern_mask=for_pattern_mask)
            else:
                cond_mask = get_randmask(ob_mask_t)

        s = {
            "observed_data": ob_data,
            "observed_mask": ob_mask,
            "gt_mask": gt_mask,
            "hist_mask": for_pattern_mask,
            "timepoints": np.arange(self.eval_length),
            "cut_length": self.cut_length[org_index],
            "cond_mask": cond_mask.numpy(),
        }

        return s

    def __len__(self):
        return len(self.use_index)


def get_dataloader_Tianjin(device, seed=1, target_strategy="random", batch_size=16, missing_ratio=0.1, validindex=0, missing_pattern=None):
    dataset = AirQuality_Dataset(mode="train", target_strategy=target_strategy, validindex=validindex, missing_ratio=missing_ratio, seed=seed)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=True
    )
    dataset_test = AirQuality_Dataset(mode="test", target_strategy=target_strategy, validindex=validindex, missing_ratio=missing_ratio, seed=seed)
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=1, shuffle=False
    )
    dataset_valid = AirQuality_Dataset(mode="valid", target_strategy=target_strategy, validindex=validindex, missing_ratio=missing_ratio, seed=seed)
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=1, shuffle=False
    )

    scaler = torch.from_numpy(dataset.std).to(device).float()
    mean_scaler = torch.from_numpy(dataset.mean).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler