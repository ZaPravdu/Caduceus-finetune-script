import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

from transformers import AutoModelForMaskedLM,AutoTokenizer
from pytorch_lightning import LightningModule

from einops.layers.torch import Rearrange
import torch

import torch.distributed as dist
from caduceus import CaduceusConfig, CaduceusForMaskedLM

class HyperGeneModelClass(LightningModule):
    """
    This is a hyper class for gene expression level task. It is used to evaluate Xpresso. See https://www.cell.com/cell-reports/pdfExtended/S2211-1247(20)30616-1
    """
    def __init__(self, lr=0.0001):
        super(HyperGeneModelClass,self).__init__()
        self.lr=lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = F.mse_loss(y_hat, y)
        loss=poisson_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True,on_epoch=True, prog_bar=True,sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        return {'val_loss': loss}
    def test_step(self, batch, batch_idx):
        data, target=batch
        output=self(data)
        self.outputs.append(output.flatten(0).detach().cpu().numpy())
        self.targets.append(target.flatten(0).detach().cpu().numpy())

    def on_test_start(self) -> None:
        self.outputs=[]
        self.targets=[]

    def on_test_end(self) -> None:
        self.outputs = np.concatenate(self.outputs, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

        pr, p = self.calculate_pearsonr(self.outputs, self.targets)
        if self.logger is not None:
            self.logger.experiment.log(dict(pearsonR=pr))
        print(f'Test PearsonR: {pr:.2f}')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def calculate_pearsonr(self, x, y):
        # 计算Pearson相关系数及其p-value
        correlation, p_value = pearsonr(x, y)
        return correlation, p_value

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# losses and metrics

def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()
class HyperEPIModelClass(HyperGeneModelClass):
    """
    This is a hyper class for epi task
    """

    def __init__(self, lr=0.0001):
        super(HyperEPIModelClass,self).__init__()
        self.lr=lr

    def on_test_start(self) -> None:
        self.min_DNase_pr= 0
        self.min_CAGE_pr = 0
        self.min_H3K27ac_pr = 0
        self.min_H3K4me3_pr = 0

        self.df=pd.DataFrame(
            {
                'DNase': [],
                'CAGE': [],
                'H3K27ac': [],
                'H3K4me3': [],
            }
        )
    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)

        pr_DNase, _ = self.calculate_pearsonr(output[:,:,0].flatten(0).detach().cpu().numpy() , target[:,:,0].flatten(0).detach().cpu().numpy())
        pr_CAGE, _ = self.calculate_pearsonr(output[:,:,1].flatten(0).detach().cpu().numpy(), target[:,:,1].flatten(0).detach().cpu().numpy())
        pr_H3K27ac, _ = self.calculate_pearsonr(output[:,:,2].flatten(0).detach().cpu().numpy(), target[:,:,2].flatten(0).detach().cpu().numpy())
        pr_H3K4me3, _ = self.calculate_pearsonr(output[:,:,3].flatten(0).detach().cpu().numpy(), target[:,:,3].flatten(0).detach().cpu().numpy())

        temp = pd.DataFrame({
            'DNase': [pr_DNase],
            'CAGE': [pr_CAGE],
            'H3K27ac': [pr_H3K27ac],
            'H3K4me3': [pr_H3K4me3],
        })
        self.df = pd.concat([self.df, temp])

        if np.isnan(pr_DNase):
            pr_DNase=0
        if np.isnan(pr_CAGE):
            pr_CAGE=0
        if np.isnan(pr_H3K27ac):
            pr_H3K27ac=0
        if np.isnan(pr_H3K4me3):
            pr_H3K4me3=0

        self.min_DNase_pr += pr_DNase
        self.min_CAGE_pr += pr_CAGE
        self.min_H3K27ac_pr += pr_H3K27ac
        self.min_H3K4me3_pr += pr_H3K4me3

        # data=[pr_DNase,pr_CAGE,pr_H3K27ac,pr_H3K4me3]

    def on_test_end(self) -> None:
        # a csv for evaluation of distribution of performance on test set.
        self.df.to_csv('./caduceus_fold1_samples_relevance.csv')

        self.min_DNase_pr/= self.test_length
        self.min_CAGE_pr /= self.test_length
        self.min_H3K27ac_pr /= self.test_length
        self.min_H3K4me3_pr /= self.test_length

        print(f'Test PearsonR DNase: {self.min_DNase_pr:.2f}')
        print(f'Test PearsonR CAGE: {self.min_CAGE_pr:.2f}')
        print(f'Test PearsonR H3K27ac: {self.min_H3K27ac_pr:.2f}')
        print(f'Test PearsonR H3K4me3: {self.min_H3K4me3_pr:.2f}')

        if self.logger is not None:
                self.logger.experiment.log(dict(pearsonR_DNase=self.min_DNase_pr, pearsonR_CAGE=self.min_CAGE_pr,
                                                pearsonR_H3K27ac=self.min_H3K27ac_pr,pearsonR_H3K4me3e=self.min_H3K4me3_pr))


class CaduceusEPIModel(HyperEPIModelClass):
    """
    This class is used to finetune Caduceus for epi task.
    """
    def __init__(self, model_name, output_hidden_states=False, linear_probe=True, config=None):
        super(CaduceusEPIModel, self).__init__()
        self.model_name = model_name

        if config:
            config = CaduceusConfig(**config)
            self.backbone = CaduceusForMaskedLM(config)

        else:
            self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name,
                                                                 trust_remote_code=True,
                                                                 output_hidden_states=output_hidden_states)

        if linear_probe:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.att_pool=AttentionPool(256,64)
        # the same output layer used in enformer model
        self.final_pointwise = nn.Sequential(
            ConvBlock(256, 128, 1),
            nn.Dropout(0.05),
            GELU()
        )

        self.regressor=nn.Conv1d(128,4, kernel_size=1)

        # for m in self.projector.modules():
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_normal_(m.weight)
        #
        # for m in self.regressor.modules():
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_normal_(m.weight)


    def forward(self, x):
        x = self.backbone(x)

        x=x['hidden_states']
        x = torch.stack(x, dim=-1)

        x = torch.sum(x, dim=-1)
        x=x.transpose(1,2)
        x=self.att_pool(x)
        x=self.final_pointwise(x)

        x=self.regressor(x)
        x=x.transpose(1,2)

        return x

class CaduceusGeneModel(HyperEPIModelClass):
    """
        This class is used to finetune Caduceus for gene expression level prediction task.
        """
    def __init__(self, model_name, output_hidden_states=False, linear_probe=True, config=None, in_features=256):
        super(CaduceusGeneModel, self).__init__()
        self.model_name = model_name

        if config:
            config = CaduceusConfig(**config)
            self.backbone = CaduceusForMaskedLM(config)

        else:
            self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name,
                                                                 trust_remote_code=True,
                                                                 output_hidden_states=output_hidden_states)

        if linear_probe:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Flatten(-2, -1),
            nn.Linear(20000, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 1),
        )

        # for m in self.projector.modules():
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_normal_(m.weight)
        #
        # for m in self.regressor.modules():
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_normal_(m.weight)


    def forward(self, x):
        x = self.backbone(x)
        x=x['hidden_states']
        x = torch.stack(x, dim=-1)

        x = torch.sum(x, dim=-1) # sum all hidden states
        x=x.transpose(1,2)

        x = self.projector(x)
        x = self.regressor(x)

        return x

class CNNEPIModel(HyperEPIModelClass):
    """
    This is a Xpresso-like model to be tested as a baseline on epi task.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(4, 128, kernel_size=7)
        self.max_pool = nn.MaxPool1d(kernel_size=8, stride=8)
        self.conv2 = ConvBlock(128, 128, kernel_size=10)

        self.final_pointwise = nn.Sequential(
            ConvBlock(128, 128, 1),
            nn.Dropout(0.05),
            GELU()
        )

        self.regressor = nn.Conv1d(128, 4, kernel_size=1, padding='same')

    def forward(self, x):
        x = self.seq2onehot(x)
        x = x.transpose(1, 2)

        x = self.conv1(x)  # n, dim, seq
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)  # n, dim, seq/64

        x = self.final_pointwise(x)

        x = self.regressor(x)
        x = x.transpose(1, 2)
        return x

    def seq2onehot(self, x):
        x -= 7
        x = F.one_hot(x, num_classes=5)
        x = x[..., :4]
        return x.float()

class CNNGeneModel(HyperGeneModelClass):
    """
    This is a Xpresso-like model to be tested as a baseline.
    """
    def __init__(self):
        super().__init__()

        self.conv1=ConvBlock(4,128, kernel_size=7)
        self.max_pool=nn.MaxPool1d(kernel_size=8,stride=8)
        self.conv2=ConvBlock(128,128, kernel_size=10)

        self.final_pointwise = nn.Sequential(
            ConvBlock(128, 128, 1),
            nn.Dropout(0.05),
            GELU()
        )

        self.regressor=nn.Linear(40064,1)

    def forward(self,x):
        x = self.seq2onehot(x)
        x=x.transpose(1,2)

        x=self.conv1(x) # n, dim, seq
        x = self.max_pool(x)
        x=self.conv2(x)
        x=self.max_pool(x)# n, dim, seq/64

        x = self.final_pointwise(x)
        x=self.regressor(x)
        x = x.transpose(1, 2)
        return x

    def seq2onehot(self, x):
        x -= 7
        x = F.one_hot(x, num_classes=5)
        x = x[..., :4]
        return x.float()

# class ConvBlock(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super().__init__()
#         self.module=nn.Sequential(
#             nn.Conv1d(in_channel, out_channel,kernel_size=3, padding='same'),
#             nn.ReLU(),
#             nn.BatchNorm1d(out_channel)
#         )
#         self.bypass=nn.Sequential(
#             nn.Conv1d(in_channel, out_channel, kernel_size=1),
#             nn.ReLU(),
#             nn.BatchNorm1d(out_channel)
#         )
#         torch.nn.init.kaiming_normal(self.module[0].weight)
#
#     def forward(self,x):
#         bypass=self.bypass(x)
#         x=self.module(x)
#         x+=bypass
#         return x


class BPNet(HyperGeneModelClass):
    """A basic BPNet model with stranded profile and total count prediction.

    This is a reference implementation for BPNet models. It exactly matches the
    architecture in the official ChromBPNet repository. It is very similar to
    the implementation in the official basepairmodels repository but differs in
    when the activation function is applied for the resifual layers. See the
    BasePairNet object below for an implementation that matches that repository.

    The model takes in one-hot encoded sequence, runs it through:

    (1) a single wide convolution operation

    THEN

    (2) a user-defined number of dilated residual convolutions

    THEN

    (3a) profile predictions done using a very wide convolution layer
    that also takes in stranded control tracks

    AND

    (3b) total count prediction done using an average pooling on the output
    from 2 followed by concatenation with the log1p of the sum of the
    stranded control tracks and then run through a dense layer.

    This implementation differs from the original BPNet implementation in
    two ways:

    (1) The model concatenates stranded control tracks for profile
    prediction as opposed to adding the two strands together and also then
    smoothing that track

    (2) The control input for the count prediction task is the log1p of
    the strand-wise sum of the control tracks, as opposed to the raw
    counts themselves.

    (3) A single log softmax is applied across both strands such that
    the logsumexp of both strands together is 0. Put another way, the
    two strands are concatenated together, a log softmax is applied,
    and the MNLL loss is calculated on the concatenation.

    (4) The count prediction task is predicting the total counts across
    both strands. The counts are then distributed across strands according
    to the single log softmax from 3.


    Parameters
    ----------
    n_filters: int, optional
        The number of filters to use per convolution. Default is 64.

    n_layers: int, optional
        The number of dilated residual layers to include in the model.
        Default is 8.

    n_outputs: int, optional
        The number of profile outputs from the model. Generally either 1 or 2
        depending on if the data is unstranded or stranded. Default is 2.

    n_control_tracks: int, optional
        The number of control tracks to feed into the model. When predicting
        TFs, this is usually 2. When predicting accessibility, this is usualy
        0. When 0, this input is removed from the model. Default is 2.

    alpha: float, optional
        The weight to put on the count loss.

    profile_output_bias: bool, optional
        Whether to include a bias term in the final profile convolution.
        Removing this term can help with attribution stability and will usually
        not affect performance. Default is True.

    count_output_bias: bool, optional
        Whether to include a bias term in the linear layer used to predict
        counts. Removing this term can help with attribution stability but
        may affect performance. Default is True.

    name: str or None, optional
        The name to save the model to during training.

    trimming: int or None, optional
        The amount to trim from both sides of the input window to get the
        output window. This value is removed from both sides, so the total
        number of positions removed is 2*trimming.

    verbose: bool, optional
        Whether to display statistics during training. Setting this to False
        will still save the file at the end, but does not print anything to
        screen during training. Default is True.
    """

    def __init__(self, n_filters=64, n_layers=8, n_outputs=2,
                 n_control_tracks=2, alpha=1, profile_output_bias=True,
                 count_output_bias=True, name=None, trimming=None, verbose=True):
        super(BPNet, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.n_control_tracks = n_control_tracks

        self.alpha = alpha
        self.name = name or "bpnet.{}.{}".format(n_filters, n_layers)
        self.trimming = trimming or 2 ** n_layers

        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
        self.irelu = torch.nn.ReLU()

        self.rconvs = torch.nn.ModuleList([
            torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2 ** i,
                            dilation=2 ** i) for i in range(1, self.n_layers + 1)
        ])
        self.rrelus = torch.nn.ModuleList([
            torch.nn.ReLU() for i in range(1, self.n_layers + 1)
        ])

        # self.fconv = torch.nn.Conv1d(n_filters, n_outputs,
        #                              kernel_size=75, padding=37, bias=profile_output_bias)

        n_count_control = 1 if n_control_tracks > 0 else 0
        self.linear = torch.nn.Linear(n_filters, 1,
                                      bias=count_output_bias)

        self.norm=nn.LayerNorm([1,20000])



    def forward(self, X, X_ctl=None):
        """A forward pass of the model.

        This method takes in a nucleotide sequence X, a corresponding
        per-position value from a control track, and a per-locus value
        from the control track and makes predictions for the profile
        and for the counts. This per-locus value is usually the
        log(sum(X_ctl_profile)+1) when the control is an experimental
        read track but can also be the output from another model.

        Parameters
        ----------
        X: torch.tensora, shape=(batch_size, 4, length)
            The one-hot encoded batch of sequences.

        X_ctl: torch.tensor or None, shape=(batch_size, n_strands, length)
            A value representing the signal of the control at each position in
            the sequence. If no controls, pass in None. Default is None.

        Returns
        -------
        y_profile: torch.tensor, shape=(batch_size, n_strands, out_length)
            The output predictions for each strand trimmed to the output
            length.
        """
        # X=X.unsqueeze(1).float()
        # X=self.norm(X)

        start, end = self.trimming, X.shape[2] - self.trimming

        X = self.irelu(self.iconv(X))
        for i in range(self.n_layers):
            X_conv = self.rrelus[i](self.rconvs[i](X))
            X = torch.add(X, X_conv)

        # if X_ctl is None:
        #     X_w_ctl = X
        # else:
        #     X_w_ctl = torch.cat([X, X_ctl], dim=1)

        # y_profile = self.fconv(X_w_ctl)[:, :, start:end]

        # counts prediction
        X = torch.mean(X[:, :, start - 37:end + 37], dim=2)
        if X_ctl is not None:
            X_ctl = torch.sum(X_ctl[:, :, start - 37:end + 37], dim=(1, 2))
            X_ctl = X_ctl.unsqueeze(-1)
            X = torch.cat([X, torch.log(X_ctl + 1)], dim=-1)

        y_counts = self.linear(X).reshape(X.shape[0], 1)
        return  y_counts

class _Exp(torch.nn.Module):
    def __init__(self):
        super(_Exp, self).__init__()

    def forward(self, X):
        return torch.exp(X)


class _Log(torch.nn.Module):
    def __init__(self):
        super(_Log, self).__init__()

    def forward(self, X):
        return torch.log(X)
class ChromBPNet(torch.nn.Module):
    """A ChromBPNet model.

    ChromBPNet is an extension of BPNet to handle chromatin accessibility data,
    in contrast to the protein binding data that BPNet handles. The distinction
    between these data types is that an enzyme used in DNase-seq and ATAC-seq
    experiments itself has a soft sequence preference, meaning that the
    strength of the signal is driven by real biology but that the exact read
    mapping locations are driven by the soft sequence bias of the enzyme.

    ChromBPNet handles this by treating the data using two models: a bias
    model that is initially trained on background (non-peak) regions where
    the bias dominates, and an accessibility model that is subsequently trained
    using a frozen version of the bias model. The bias model learns to remove
    the enzyme bias so that the accessibility model can learn real motifs.


    Parameters
    ----------
    bias: torch.nn.Module
        This model takes in sequence and outputs the shape one would expect in
        ATAC-seq data due to Tn5 bias alone. This is usually a BPNet model
        from the bpnet-lite repo that has been trained on GC-matched non-peak
        regions.

    accessibility: torch.nn.Module
        This model takes in sequence and outputs the accessibility one would
        expect due to the components of the sequence, but also takes in a cell
        representation which modifies the parameters of the model, hence,
        "dynamic." This model is usually a DynamicBPNet model, defined below.

    name: str
        The name to prepend when saving the file.
    """

    def __init__(self, bias, accessibility, name):
        super(ChromBPNet, self).__init__()
        for parameter in bias.parameters():
            parameter.requires_grad = False

        self.bias = bias
        self.accessibility = accessibility
        self.name = name
        self.logger = None
        self.n_control_tracks = accessibility.n_control_tracks
        self.n_outputs = 1
        self._log = _Log()
        self._exp1 = _Exp()
        self._exp2 = _Exp()

    def forward(self, X, X_ctl=None):
        """A forward pass through the network.

        This function is usually accessed through calling the model, e.g.
        doing `model(x)`. The method defines how inputs are transformed into
        the outputs through interactions with each of the layers.


        Parameters
        ----------
        X: torch.tensor, shape=(-1, 4, 2114)
            A one-hot encoded sequence tensor.

        X_ctl: ignore
            An ignored parameter for consistency with attribution functions.


        Returns
        -------
        y_profile: torch.tensor, shape=(-1, 1000)
            The predicted logit profile for each example. Note that this is not
            a normalized value.
        """

        acc_profile, acc_counts = self.accessibility(X)
        bias_profile, bias_counts = self.bias(X)

        y_profile = acc_profile + bias_profile
        y_counts = self._log(self._exp1(acc_counts) + self._exp2(bias_counts))

        return y_profile, y_counts

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)

        return (x * attn).sum(dim = -1)

class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]

def ConvBlock(dim, dim_out = None, kernel_size = 1, is_distributed = None):
    batchnorm_klass = MaybeSyncBatchnorm(is_distributed = is_distributed)

    return nn.Sequential(
        batchnorm_klass(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )

def MaybeSyncBatchnorm(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d

def exists(val):
    return val is not None
#
def default(val, d):
    return val if exists(val) else d
