import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import pickle
import torch


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


def get_block_mask(observed_mask, train_missing_pattern='block'):
    rand_sensor_mask = torch.rand_like(observed_mask)
    randint = np.random.randint
    sample_ratio = np.random.rand()
    sample_ratio = sample_ratio * 0.15
    mask = rand_sensor_mask < sample_ratio
    min_seq = 12
    max_seq = 24
    for col in range(observed_mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, observed_mask.shape[0] - 1)
        mask[idxs, col] = True
    rand_base_mask = torch.rand_like(observed_mask) < 0.05
    reverse_mask = mask | rand_base_mask
    block_mask = 1 - reverse_mask.to(torch.float32)

    cond_mask = observed_mask.clone()
    mask_choice = np.random.rand()
    cond_mask = block_mask * cond_mask

    return cond_mask

def sample_mask(shape, p=0.0015, p_noise=0.05, max_seq=1, min_seq=1, rng=None):
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers
    mask = rand(shape) < p 
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col]) 
        if not len(idxs): 
            continue
        fault_len = min_seq 
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq)) 
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs]) 
        idxs = np.unique(idxs_ext) 
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True 
    mask = mask | (rand(mask.shape) < p_noise) 

    return mask.astype('uint8')


class Discharge_Dataset(Dataset):
    def __init__(self, seed=1, eval_length=16, mode="train", val_len=0.1, test_len=0.2, eval_missing_pattern='block',
                train_missing_pattern='point'):
        self.eval_length = eval_length
        self.train_missing_pattern = train_missing_pattern
        self.mode = mode
        self.use_index = []
        self.cut_length = []

        df = pd.read_csv('./data/hydrology/SSC_discharge.csv', index_col=0)      
        df.index = pd.to_datetime(df.index)
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='1D')
        df = df.reindex(index=date_range) 

        start_date = '2015/4/15'
        end_date = '2022/9/30'
        
        #### log transformation #####
        df = df.loc[start_date:end_date, :] # 特定时间范围的数据
        ob_mask = ~np.isnan(df.values)
        df.fillna(method='ffill', axis=0, inplace=True)

        # SEED = 45678
        SEED = seed
        self.rng = np.random.default_rng(SEED)
        if eval_missing_pattern == 'block':
            eval_mask = sample_mask(shape=(2726, 20), p=0.0015, p_noise=0.05, min_seq=12, max_seq=12 * 4, rng=self.rng)
            print(eval_mask[:5])
        elif eval_missing_pattern == 'point':
            eval_mask = sample_mask(shape=(2726, 20), p=0., p_noise=0.25, max_seq=12, min_seq=12 * 4, rng=self.rng)
        gt_mask = (1-(eval_mask | (1-ob_mask))).astype('uint8')

        self.train_mean = np.zeros(20)
        self.train_std = np.zeros(20)
        for k in range(20):
            tmp_data = df.iloc[:, k][ob_mask[:, k] == 1] 
            self.train_mean[k] = tmp_data.mean()
            self.train_std[k] = tmp_data.std()
        path = "./data/hydrology/discharge_meanstd.pk"
        with open(path, "wb") as f:
            pickle.dump([self.train_mean, self.train_std], f)

        val_start = int((1 - val_len - test_len) * len(df))
        test_start = int((1 - test_len) * len(df))
        c_data = (
             (df.fillna(0).values - self.train_mean) / self.train_std
        ) * ob_mask
        if mode == 'train':
            self.observed_mask = ob_mask[:val_start]
            self.gt_mask = gt_mask[:val_start]
            self.observed_data = c_data[:val_start]
        elif mode == 'valid':
            self.observed_mask = ob_mask[val_start: test_start]
            self.gt_mask = gt_mask[val_start: test_start]
            self.observed_data = c_data[val_start: test_start]
        elif mode == 'test':
            self.observed_mask = ob_mask[test_start:]
            self.gt_mask = gt_mask[test_start:]
            self.observed_data = c_data[test_start:]
        current_length = len(self.observed_mask) - eval_length + 1

        if mode == "test":
            n_sample = len(self.observed_data) // eval_length
            c_index = np.arange(
                0, 0 + eval_length * n_sample, eval_length
            )
            self.use_index += c_index.tolist()
            self.cut_length += [0] * len(c_index)
            if len(self.observed_data) % eval_length != 0:
                self.use_index += [current_length - 1]
                self.cut_length += [eval_length - len(self.observed_data) % eval_length]
        elif mode != "test":
            self.use_index = np.arange(current_length)
            self.cut_length = [0] * len(self.use_index)

    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        ob_data = self.observed_data[index: index + self.eval_length]
        ob_mask = self.observed_mask[index: index + self.eval_length]
        ob_mask_t = torch.tensor(ob_mask).float()
        gt_mask = self.gt_mask[index: index + self.eval_length]
        if self.mode != 'train':
            cond_mask = torch.tensor(gt_mask).to(torch.float32)
        else:
            if self.train_missing_pattern != 'point':
                cond_mask = get_block_mask(ob_mask_t, train_missing_pattern=self.train_missing_pattern)
            else:
                cond_mask = get_randmask(ob_mask_t)
        domain_indicator = torch.tensor([1])
        s = {
            "observed_data": ob_data,
            "observed_mask": ob_mask,
            "gt_mask": gt_mask,
            "timepoints": np.arange(self.eval_length),
            "cut_length": self.cut_length[org_index],
            "cond_mask": cond_mask,
            "domain_indicator": domain_indicator
        }
        return s

    def __len__(self):
        return len(self.use_index)


def get_dataloader_discharge(batch_size, device, seed=1, val_len=0.1, num_workers=4, 
                   eval_missing_pattern='block', train_missing_pattern='block'):
    dataset = Discharge_Dataset(seed=seed, mode="train", eval_missing_pattern=eval_missing_pattern,
                             train_missing_pattern=train_missing_pattern)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    dataset_test = Discharge_Dataset(seed=seed, mode="test", eval_missing_pattern=eval_missing_pattern,
                             train_missing_pattern=train_missing_pattern)
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    dataset_valid = Discharge_Dataset(seed=seed, mode="valid", val_len=val_len, eval_missing_pattern=eval_missing_pattern,
                             train_missing_pattern=train_missing_pattern)
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    scaler = torch.from_numpy(dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler