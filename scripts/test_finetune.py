import caduceus_finetune as cfx
import caduceus_finetune.utils as utils
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer

model_name="kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
testset=utils.EPIDataset('../data/K562_EPI/K562_EPI_fold1_ensembl_test.csv', model_name=model_name)
# testset=Subset(testset,range(20))

# model=cfx.CNNSeqModel().load_from_checkpoint('./weight/K562-fold1-CNN-epi/gdscnn57/checkpoints/epoch=0-step=3267.ckpt')
model=cfx.CaduceusEPIModel.load_from_checkpoint(
    '../weight/K562-fold1-Ph-epi/gou6ln3o/checkpoints/epoch=0-step=3267.ckpt',
    model_name=model_name,
    output_hidden_states=True)

for p in model.parameters():
    p.requires_grad=False

trainer = Trainer( max_epochs=1,accelerator='gpu',log_every_n_steps=1, logger=None)
model.test_length=len(testset)
testloader= DataLoader(testset,batch_size=1, drop_last=True, num_workers=16)
trainer.test(model, testloader)




