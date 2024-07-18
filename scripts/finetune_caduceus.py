import caduceus_finetune as cf
import caduceus_finetune.utils as utils
from torch.utils.data import DataLoader, Subset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# experiment_name='GM12878-fold1-Ph-en'
experiment_name = 'test'
epochs = 1
lr = 0.0001
batch_size = 4
cell = 'GM12878'
cell = 'Actual_' + cell

# See the `Caduceus` collection page on the hub for list of available models.
model_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"

model = cf.CaduceusGeneModel(model_name, output_hidden_states=True, in_features=256)
# model =cfx.BPNet()
# model=cfx.CNNSeqModel()
# dataset=utils.HG38Dataset('~/caduceus/data/Xpresso/Xpresso_train.csv')
trainset = utils.GeneDataset('/data/Xpresso/Xpresso_fold1_ensembl_train.csv', cell=cell,
                             model_name=model_name)
validset = utils.GeneDataset('/data/Xpresso/Xpresso_fold1_ensembl_valid.csv', cell=cell,
                             model_name=model_name)
testset = utils.GeneDataset('/data/Xpresso/Xpresso_fold1_ensembl_test.csv', cell=cell, model_name=model_name)

if experiment_name == 'test':
    trainset = Subset(trainset, range(20))
    validset = Subset(validset, range(20))
    testset = Subset(testset, range(20))
    wandb_logger = None
else:
    wandb_logger = WandbLogger(project=experiment_name, save_dir='../weight/')

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # 监控验证集损失
    dirpath=f'./weight/{experiment_name}',
    filename='{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min'
)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
trainer = Trainer(callbacks=[early_stopping, checkpoint_callback], max_epochs=1, accelerator='gpu', logger=wandb_logger,
                  default_root_dir=f'./weight/{experiment_name}', log_every_n_steps=1)

model.cuda()

for e in range(epochs):
    # 创建DataLoader
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)
    validloader = DataLoader(validset, batch_size=batch_size, drop_last=True, num_workers=16)

    trainer.fit(model, trainloader, validloader)

testloader = DataLoader(testset, batch_size=batch_size, drop_last=True, num_workers=16)
trainer.test(model, testloader)

print()
