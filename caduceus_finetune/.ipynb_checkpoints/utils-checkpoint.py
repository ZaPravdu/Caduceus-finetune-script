import torch
import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path
from pyfaidx import Fasta
import sys
import numpy as np

from src.dataloaders.utils.mlm import mlm_getitem
from src.dataloaders.utils.rc import coin_flip, string_reverse_complement
from pytorch_lightning.callbacks import TQDMProgressBar
import torch.nn.functional as F
import math

import pandas as pd
import pyranges as pr
import pyBigWig

from kipoiseq import Interval
import pyfaidx
import kipoiseq
import numpy as np
import pyranges as pr
from Bio.Seq import Seq
from tqdm import tqdm
import ast

MAX_ALLOWED_LENGTH = 2 ** 20

class FastaInterval:
    """Retrieves sequences from a fasta file given a chromosome and start/end indices."""
    def __init__(
            self,
            *,
            fasta_file,
            return_seq_indices=False,
            rc_aug=False,
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), "Path to fasta file must exist!"

        self.seqs = Fasta(str(fasta_file))
        self.return_seq_indices = return_seq_indices
        self.rc_aug = rc_aug

        # calc len of each chromosome in fasta file, store in dict
        self.chr_lens = {}

        for chr_name in self.seqs.keys():
            self.chr_lens[chr_name] = len(self.seqs[chr_name])

    @staticmethod
    def _compute_interval(start, end, max_length, i_shift):
        if max_length == MAX_ALLOWED_LENGTH:
            return start, end
        if max_length < MAX_ALLOWED_LENGTH:
            assert MAX_ALLOWED_LENGTH % max_length == 0
            return start + i_shift * max_length, start + (i_shift + 1) * max_length
        else:
            raise ValueError(f"`max_length` {max_length} (> 2^{int(math.log(MAX_ALLOWED_LENGTH, 2))}) is too large!")

    def __call__(
            self,
            chr_name,
            start,
            end,
            max_length,
            i_shift,
            return_augs=False,
    ):
        """
        max_length passed from dataset, not from init
        """
        chromosome = self.seqs[chr_name]
        chromosome_length = self.chr_lens[chr_name]

        start, end = self._compute_interval(start, end, max_length, i_shift)

        if end > chromosome_length:
            # Shift interval down
            start = start - (end - chromosome_length)
            end = chromosome_length
            assert start == chromosome_length - max_length

        if start < 0:
            # Shift interval up
            end = end - start
            start = 0
            assert end == max_length

        if end > chromosome_length:
            # This may occur if start + MAX_ALLOWED_LENGTH extends beyond the end of the chromosome
            start = chromosome_length - max_length
            end = chromosome_length

        seq = str(chromosome[start:end])

        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)

        return seq

class HG38Dataset(torch.utils.data.Dataset):
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

        # replace N token with a pad token, so we can ignore it in the loss
        # seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int["N"], self.tokenizer.pad_token_id)
        # if self.one_hot:
        #     seq-=7
        #     abnorm= torch.where(seq>3)
        #     seq[abnorm]=3
        #     seq = F.one_hot(seq , num_classes=4).float()
        #     seq[abnorm] = 0
        #     seq = seq.transpose(0, 1)

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
        self.data = pd.read_csv(csv_path)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        """Returns a sequence of specified len"""
        # sample a random row from df
        row=self.data.iloc[idx]
        DNase_profile=np.array(ast.literal_eval(row['DNase'])).astype('float')
        CAGE_profile=np.array(ast.literal_eval(row['CAGE'])).astype('float')
        H3K27ac_profile=np.array(ast.literal_eval(row['H3K27ac'])).astype('float')
        H3K4me3_profile=np.array(ast.literal_eval(row['H3K4me3'])).astype('float')
        seq=row['seq']

        target=np.stack([DNase_profile, CAGE_profile, H3K27ac_profile, H3K4me3_profile])

        seq = self.tokenizer(seq,truncation=True,add_special_tokens=False)

        # convert to tensor
        seq = torch.LongTensor(seq['input_ids'])

        if not isinstance(target, torch.Tensor):
            target=torch.Tensor(target)

        # replace N token with a pad token, so we can ignore it in the loss
        # seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int["N"], self.tokenizer.pad_token_id)
        # if self.one_hot:
        #     seq-=7
        #     abnorm= torch.where(seq>3)
        #     seq[abnorm]=3
        #     seq = F.one_hot(seq , num_classes=4).float()
        #     seq[abnorm] = 0
        #     seq = seq.transpose(0, 1)
        target=torch.log(target+1)
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
class MyTQDMProgressBar(TQDMProgressBar):

    def __init__(self):
        super(MyTQDMProgressBar, self).__init__()

    def init_validation_tqdm(self):
        bar = Tqdm(
            desc=self.validation_description,
            position=0,  # 这里固定写0
            disable=self.is_disabled,
            leave=True,  # leave写True
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

one_hot_embed = torch.zeros(256, 4)
one_hot_embed[ord('a')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('c')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('g')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('t')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('n')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('A')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('C')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('G')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('T')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('N')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('.')] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

def cast_list(t):
    return t if isinstance(t, list) else [t]
def torch_fromstring(seq_strs):
    batched = not isinstance(seq_strs, str)
    seq_strs = cast_list(seq_strs)
    np_seq_chrs = list(map(lambda t: np.fromstring(t, dtype = np.uint8), seq_strs))
    seq_chrs = list(map(torch.from_numpy, np_seq_chrs))
    return torch.stack(seq_chrs) if batched else seq_chrs[0]

def str_to_one_hot(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return one_hot_embed[seq_chrs.long()]

def seq_indices_to_one_hot(t, padding = -1):
    is_padding = t == padding
    t = t.clamp(min = 0)
    one_hot = F.one_hot(t, num_classes = 5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out


if __name__ == '__main__':
    testset=HG38Dataset('~/caduceus/data/Xpresso/xpresso_data_for_training_caduecus.csv')
    print(testset[0])
    print(len(testset))

