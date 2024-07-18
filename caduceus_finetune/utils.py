import torch
from transformers import AutoTokenizer
from src.dataloaders.utils.mlm import mlm_getitem
import pandas as pd
from kipoiseq import Interval
import pyfaidx
import random
import h5py
from Bio.Seq import Seq

class GeneDataset(torch.utils.data.Dataset):
    """Loop through bed file, retrieve (chr, start, end), query fasta file for sequence."""

    def __init__(
            self,
            csv_path,
            cell=None,
            tokenizer=None,
            add_eos=False,
            return_augs=False,
            model_name=None,
            mlm=False,
            mlm_probability=None,
            one_hot=False
    ):
        self.one_hot=one_hot
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.mlm=mlm
        self.mlm_probability=mlm_probability
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.cell=cell
        self.data=pd.read_csv(csv_path, usecols=[cell,'sequence_20kb_Ensembl'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        # sample a random row from df
        target,seq=self.data.iloc[idx]

        seq = self.tokenizer(seq,truncation=True,add_special_tokens=False)

        # convert to tensor
        seq = torch.LongTensor(seq['input_ids'])
        if self.mlm:
            seq, target = mlm_getitem(
                seq,
                mlm_probability=self.mlm_probability,
                contains_eos=self.add_eos,
                tokenizer=self.tokenizer,
            )

        if not isinstance(target, torch.Tensor):
            target=torch.Tensor([target])

        return seq, target

class FastaStringExtractor:
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()

class EPIDataset(torch.utils.data.Dataset):
    """
    THis is a Dataset class for epi task. I store sequences in a csv file and targets in a h5 file. You can customize this class
    """

    def __init__(
            self,
            csv_path,
            cell=None,
            tokenizer=None,
            add_eos=False,
            return_augs=False,
            model_name=None,
            mlm=False,
            mlm_probability=None,
            one_hot=False,
            data_augment=None,
    ):
        self.one_hot=one_hot
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.mlm=mlm
        self.mlm_probability=mlm_probability
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.cell=cell
        self.data = pd.read_csv(csv_path)
        self.data_augment=data_augment
        self.np_idx=self.data['Unnamed: 0.1']

        with h5py.File('./data/K562_EPI/epi.h5', 'r') as f:
            # 获取数据集
            dataset_name = 'target'
            if dataset_name in f:
                targets = f[dataset_name]

                # 将数据集内容读取为NumPy数组
                self.targets = targets[:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        """Returns a sequence of specified len"""
        # sample a random row from df
        row=self.data.iloc[idx]

        target=self.targets[self.np_idx[idx]]

        seq=row['seq']

        if not isinstance(target, torch.Tensor):
            target=torch.Tensor(target)

        if self.data_augment:
            if random.random()<self.data_augment:
                seq=str(Seq(seq).reverse_complement())
                target=target.flip([1])

        seq = self.tokenizer(seq,truncation=True,add_special_tokens=False)

        # convert to tensor
        seq = torch.LongTensor(seq['input_ids'])

        # target=torch.log(target)

        return seq, target.transpose(0,1)

    def process_signal(self, start_i, profile, bw, chrom):
        end_i = start_i + 64
        max = bw.chroms()[chrom]
        if end_i > max:
            if start_i > max:
                signal = 0
            else:
                signal = bw.stats(chrom, start_i, max, type="sum")[0]
        else:
            signal = bw.stats(chrom, start_i, end_i, type="sum")[0]

        profile.append(signal)
        return profile



