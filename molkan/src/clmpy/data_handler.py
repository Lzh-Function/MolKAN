"""
Created by Zehao Li (Takuho Ri)
Created on 2025-02-27 (Thu)  14:30:27 (+09:00)

datahandler for clmpy
Original script: clmpy[https://github.com/mizuno-group/clmpy] composed by Shumpei Nemoto
"""

from typing import Iterator, Optional, TypeVar
import argparse
import os
import sys
import psutil
from collections import defaultdict
from functools import partial
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import math
import psutil._common
import gc
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence


# original script
class BucketSampler(Sampler):
    def __init__(self,dataset,buckets=(20,150,10),shuffle=True,batch_size=512,drop_last=False):
        super().__init__(dataset)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        length = [len(v[0]) for v in dataset]
        bucket_range = np.arange(*buckets)
        
        assert isinstance(buckets,tuple)
        bmin, bmax, bstep = buckets
        assert (bmax - bmin) % bstep == 0
        buc = torch.bucketize(torch.tensor(length),torch.tensor(bucket_range),right=False)

        bucs = defaultdict(list)
        bucket_max = max(np.array(buc))
        for i,v in enumerate(buc):
            bucs[v.item()].append(i)
        _ = bucs.pop(bucket_max)
        
        self.buckets = dict()
        for bucket_size, bucket in bucs.items():
            if len(bucket) > 0:
                self.buckets[bucket_size] = torch.tensor(bucket,dtype=torch.int)
        self.__iter__()

    def __iter__(self):
        for bucket_size in self.buckets.keys():
            self.buckets[bucket_size] = self.buckets[bucket_size][torch.randperm(self.buckets[bucket_size].nelement())]

        batches = []
        for bucket in self.buckets.values():
            curr_bucket = torch.split(bucket,self.batch_size)
            if len(curr_bucket) > 1 and self.drop_last == True:
                if len(curr_bucket[-1]) < len(curr_bucket[-2]):
                    curr_bucket = curr_bucket[:-1]
            batches += curr_bucket

        self.length = len(batches)
        if self.shuffle == True:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return self.length


# BucketSampler for DistributedDataParallel
class DistributedBucketSampler(DistributedSampler):
    def __init__(self, dataset, buckets, epoch, shuffle=True, batch_size=512, drop_last=False) -> None:
        super().__init__(dataset, shuffle=True, seed=epoch)
        """
        dataset: 対象のデータセット
        buckets: (min, max, step) のタプル、例: (20, 150, 10)
        ddp_sampler: DistributedSampler のインスタンス（各プロセスに割り当てたインデックスを提供）
        shuffle: 各バケット内でシャッフルするかどうか
        batch_size: バッチサイズ
        drop_last: 最後のバッチが不完全な場合に落とすかどうか
        """
        self.buckets = buckets
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last

        # DistributedSamplerの__iter__を呼び出して、シャッフルされたインデックスを取得
        print("loading indices...")
        indices = list(super().__iter__())

        # シーケンス長を計算
        print("loadning sequence length...")
        lengths = np.array([(dataset[i][0] != 0).sum() for i in indices]).astype(np.int16)

        # バケットの範囲
        bucket_range = np.arange(*buckets)

        # 各データ点のバケットを計算
        print("bucketizing...")
        assert isinstance(self.buckets,tuple)
        bmin, bmax, bstep = self.buckets
        assert (bmax - bmin) % bstep == 0
        buc = torch.bucketize(torch.tensor(lengths),torch.tensor(bucket_range),right=False)

        # バケットごとにインデックスをグループ化
        print("grouping...")
        bucs = defaultdict(list)
        for i,v in zip(indices, buc): # 実際にdatasetから取ってきたインデックスを反映
            bucs[v.item()].append(i)
        # _ = bucs.pop(bucket_max)

        self.buckets = dict()
        for bucket_size, bucket in bucs.items():
            if len(bucket) > 0:
                self.buckets[bucket_size] = torch.tensor(bucket,dtype=torch.int)
        self.__iter__()

        print("DDPBucketSampler initialize finished.")

    def __iter__(self):
        batches = []
        for bucket in self.buckets.values():
            curr_bucket = torch.split(bucket,self.batch_size)
            if len(curr_bucket) > 1 and self.drop_last == True:
                if len(curr_bucket[-1]) < len(curr_bucket[-2]):
                    curr_bucket = curr_bucket[:-1]
            batches += curr_bucket

        self.length = len(batches)
        if self.shuffle == True:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return self.length
    
# update: 250316
# DistributedSampler for Continuous Pretraining (50% replay)
_T_co = TypeVar("_T_co", covariant=True)
class CPTDistributedSampler(Sampler[_T_co]):
    """
    Distributed Sampler for Continuous Pretraining (50% replay)
    
    Args:
        args: argparse.Namespace
        dataset: Dataset
        phase: int
        num_replicas: Optional[int] = None
        rank: Optional[int] = None
        shuffle: bool = True
        seed: int = 0
        drop_last: bool = False
    """

    def __init__(
        self,
        args: argparse.Namespace,
        dataset: Dataset,
        phase: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.args = args
        self.dataset = dataset
        self.phase = phase+1
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.num_main_data = getattr(self.args, f"train_datanum{self.phase}")
        if self.phase > 1:
            self.num_replay_data = (self.num_main_data // 2) // self.pahse-1
        else:
            self.num_replay_data = 0
        self.num_samples = self.num_main_data + (self.num_replay_data * (self.phase-1))
        """
        If the dataset length is evenly divisible by N of replicas, then there
        is no need to drop any data, since the dataset will be split equally.
        """
        if self.drop_last and self.num_samples % self.num_replicas != 0:  # type: ignore[arg-type]
            """
            Split to nearest available length that is evenly divisible.
            This is to ensure each rank receives the same amount of data when
            using this Sampler.
            """
            self.num_samples = math.ceil(
                (self.num_samples - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self.num_samples / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = []
            for i in range(self.phase):
                index_to_add = np.sum([getattr(self.args, f"train_datanum{j+1}") for j in range(i)])
                if i+1 != self.phase:
                    index = torch.randperm(n=getattr(self.args, f"train_datanum{i+1}"), generator=g)[:self.num_replay_data] + index_to_add
                    indices += index.tolist()
                else:
                    index = torch.randperm(n=getattr(self.args, f"train_datanum{i+1}"), generator=g)[:self.num_main_data] + index_to_add
                    indices += index.tolist()
        else:
            for i in range(self.phase):
                index_to_add = np.sum([getattr(self.args, f"train_datanum{j+1}") for j in range(i)])
                if i+1 != self.phase:
                    index = torch.arange(getattr(self.args, f"train_datanum{i+1}"))[:self.num_replay_data] + index_to_add
                    indices += index.tolist()
                else:
                    index = torch.arange(getattr(self.args, f"train_datanum{i+1}"))[:self.num_main_data] + index_to_add
                    indices += index.tolist()
        
        random.seed(self.seed + self.epoch)
        random.shuffle(indices)
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) > self.num_samples:
            indices = indices[:self.num_samples]
        else:
            indices += indices[: self.num_samples - len(indices)]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


def read_smiles_from_csv(path):
    for chunk in pd.read_csv(path, usecols=["input", "output"], chunksize=100000):
        for _, row in chunk.iterrows():
            yield row["input"], row["output"]

def tokenize(s, tokens):
    s = s.replace("Br","R").replace("Cl","L")
    tok = []
    while len(s) > 0:
        if len(s) >= 2 and (s[0] == "@" or s[0] == "["):
            for j in np.arange(3,0,-1):
                if s[:j] in tokens.table:
                    tok.append(s[:j])
                    s = s[j:]
                    break
        else:
            tok.append(s[0])
            s = s[1:]
    return tok

def sfl_tokenize(s, tokens):
    s = s.replace("Br","R").replace("Cl","L")
    tok = []
    char = ""
    for v in s:
        if len(char) == 0 and v != "[":
            tok.append(v)
            continue
        char += v
        if len(char) > 1:
            if v == "]":
                if char in tokens.table:
                    tok.append(char)
                else:
                    tok.append("<unk>")
                char = ""
    return tok
                
def one_hot_encoder(tokenized, tokens):
    enc = np.array([tokens.dict[v] for v in tokenized]).astype(np.int16)
    enc = np.concatenate([np.array([1]),enc,np.array([2])]).astype(np.int16)
    assert type(enc) == np.ndarray
    padding_length = 252 - len(enc)
    if padding_length > 0:
        enc = np.concatenate([enc, np.zeros(padding_length, dtype=np.int16)])
    elif padding_length < 0:
        enc = enc[:252]
    return enc

def encode_smiles(smiles_and_args):
    input, output, tokens, tok_func = smiles_and_args
    input_tokenized = tok_func(input, tokens)
    output_tokenized = tok_func(output, tokens)
    return one_hot_encoder(input_tokenized, tokens), one_hot_encoder(output_tokenized, tokens)

def seq2id(csvpath,tokens,npypath,sfl=True):
    tok_func = sfl_tokenize if sfl else tokenize

    print("--> calculating datanum")
    datanum = 0
    for chunk in pd.read_csv(csvpath, usecols=["input"], chunksize=1000000):
        datanum += len(chunk)
    print(f"datanum: {datanum}")
    
    shape = (datanum, 2, 252)
    mm_array = np.memmap(npypath, dtype=np.int16, mode='w+', shape=shape)
    mem = psutil.virtual_memory()
    print(f"memory usage: {psutil._common.bytes2human(mem.used)} ({mem.percent}%)")

    print("---> start encoding")
    args_generator = ((input, output, tokens, tok_func) for input, output in read_smiles_from_csv(csvpath))
    with ProcessPoolExecutor() as executor:
        mapped_smiles = executor.map(encode_smiles, args_generator, chunksize=10000)
        for i, (encoded_input, encoded_output) in enumerate(mapped_smiles):
            mm_array[i, 0] = encoded_input
            mm_array[i, 1] = encoded_output
            if i % 1000000 == 0:
                mm_array.flush()
                gc.collect()
    
    mm_array.flush()
    del mm_array
    gc.collect()

    return datanum

class tokens_table():
    def __init__(self,token_path):
        with open(token_path,"r") as f:
            tokens = f.read().replace("Br","R").replace("Cl","L").split("\n")
        self.table = tokens
        self.id2sm = {i:v for i,v in enumerate(tokens)}
        self.dict = {w:v for v,w in self.id2sm.items()}
        self.length = len(self.table)

class CLM_Dataset(Dataset):
    def __init__(self,csvpath,tokens,npypath,sfl):
        self.tokens = tokens
        print("--> start seq2id encoding...")
        datanum = seq2id(csvpath, tokens, npypath, sfl)
        print("--> encoding finished, .npy saved")
        self.data = np.memmap(npypath, dtype=np.int16, mode="r", shape=(datanum, 2, 252))
        print(self.data[datanum-1])
        self.datanum = datanum
        mem = psutil.virtual_memory()
        print(f"memory usage: {psutil._common.bytes2human(mem.used)} ({mem.percent}%)")

    def __len__(self):
        return self.datanum
    
    def __getitem__(self,idx):
        return self.data[idx, 0].copy(), self.data[idx, 1].copy()

class CLM_Dataset_v2(Dataset):
    def __init__(self, path, datanum, datadim):
        self.data = np.memmap(path, dtype=np.int16, mode="r", shape=(datanum, 2, datadim))
        self.datanum = datanum
    
    def __len__(self):
        return self.datanum
    
    def __getitem__(self, idx):
        return self.data[idx, 0].copy(), self.data[idx, 1].copy()

class Encoder_Dataset(Dataset):
    def __init__(self,x,token,memmapfile,sfl):
        self.tokens = token
        self.input = seq2id(x,self.tokens,memmapfile,sfl)
        self.datanum = len(x)
    
    def __len__(self):
        return self.datanum
    
    def __getitem__(self,idx):
        out_i = self.input[idx]
        return out_i

def collate(batch):
    maxlen_x = max((x != 0).sum() for x, _ in batch)
    maxlen_y = max((y != 0).sum() for _, y in batch)
    xs = pad_sequence([torch.LongTensor(x[:maxlen_x]) for x, _ in batch],
                      batch_first=False,padding_value=0) # no padding, just list2tensor and transpose
    ys = pad_sequence([torch.LongTensor(y[:maxlen_y]) for _, y in batch],
                      batch_first=False,padding_value=0)
    return xs, ys

def encoder_collate(batch):
    xs = [torch.LongTensor(x) for x, _ in batch]
    xs = pad_sequence(xs,batch_first=False,padding_value=0)
    return xs