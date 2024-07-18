import os

import caduceus_finetune as cf
import caduceus_finetune.utils as utils
from torch.utils.data import DataLoader, Subset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

cell='K562'
# experiment_name=cell+'-fold1-Caduceus-256+4-epi-corrected'
experiment_name='test'
epochs=1
lr=0.0001
batch_size=4

cell='Actual_'+cell
config=dict(d_model=256, n_layer=4, vocab_size=12,output_hidden_states=True)

# See the `Caduceus` collection page on the hub for list of available models.
model_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"

# model = cfx.CaduceusEPIModel.load_from_checkpoint('./weight/K562-fold1-Caduceus-256+4-epi-corrected/iu4h5b2v/checkpoints/epoch=4-step=16335.ckpt',
#                                                   model_name=model_name,config=config, output_hidden_states=True)
# model = cfx.CaduceusEPIModel(model_name, output_hidden_states=True, linear_probe=False, config=config)
model=cf.CNNEPIModel()

trainset=utils.EPIDataset('./data/K562_EPI/K562_EPI_fold1_ensembl_train.csv', model_name=model_name, data_augment=0.5)
validset=utils.EPIDataset('./data/K562_EPI/K562_EPI_fold1_ensembl_valid.csv', model_name=model_name)
testset=utils.EPIDataset('./data/K562_EPI/K562_EPI_fold1_ensembl_test.csv', model_name=model_name)

if experiment_name=='test':
    trainset=Subset(trainset,range(20))
    validset=Subset(validset,range(20))
    testset=Subset(testset,range(20))
    wandb_logger = None
else:
    wandb_logger=WandbLogger(project=experiment_name, save_dir='../weight/')

checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # 监控验证集损失
        dirpath=f'./weight/{experiment_name}',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
trainer = Trainer(callbacks=[early_stopping,checkpoint_callback], max_epochs=epochs,accelerator='gpu', logger=wandb_logger, default_root_dir=f'./weight/{experiment_name}',log_every_n_steps=1)

model.cuda()

# DataLoader
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=16)
validloader = DataLoader(validset, batch_size=batch_size,drop_last=True, num_workers=16)

trainer.fit(model, trainloader, validloader)

model.test_length=len(testset)
testloader= DataLoader(testset,batch_size=1, drop_last=True, num_workers=16)
trainer.test(model, testloader)


print()

