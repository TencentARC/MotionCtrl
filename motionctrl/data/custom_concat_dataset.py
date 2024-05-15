
import bisect
import random

from torch.utils.data import Dataset, IterableDataset

from motionctrl.utils.util import instantiate_from_config


class ConcatDataset(Dataset):
    def __init__(self, dataset_configs, dataset_probs,):
        super().__init__()

        assert len(dataset_configs) == len(dataset_probs), "dataset configs and prob must have the same length"
        self.datasets = []
        for dataset_config in dataset_configs:
            dataset = instantiate_from_config(dataset_config)
            assert not isinstance(dataset, IterableDataset), "ConcatDataset does not support IterableDataset"
            self.datasets.append(dataset)

        self.dataset_probs = dataset_probs
        max_probs = max(self.dataset_probs)
        self.max_probs_idx = self.dataset_probs.index(max_probs)
        self.max_len = len(self.datasets[self.max_probs_idx])
        self.extra_scale = []
        self.required_len = []
        self.org_len = []
        for i in range(len(self.datasets)):
            self.org_len.append(len(self.datasets[i]))
            require_len = int(self.max_len * self.dataset_probs[i] / max_probs)
            if require_len < len(self.datasets[i]):
                self.required_len.append(require_len)
            else:
                self.required_len.append(len(self.datasets[i]))
            self.extra_scale.append(len(self.datasets[i]) / self.required_len[i])
            
        self.cumulative_sizes = self.cumsum(self.required_len)
        print(f"==========cumulative_sizes={self.cumulative_sizes}==========")

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = e
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1] 
        
        if self.extra_scale[dataset_idx] > 1:
            min_sample_idx = int(sample_idx * self.extra_scale[dataset_idx])
            max_sample_idx = int((sample_idx + 1) * self.extra_scale[dataset_idx])
            if max_sample_idx >= self.org_len[dataset_idx]:
                max_sample_idx = self.org_len[dataset_idx] - 1
            if max_sample_idx > min_sample_idx:
                sample_idx = random.randint(min_sample_idx, max_sample_idx)

        return self.datasets[dataset_idx][sample_idx]            

