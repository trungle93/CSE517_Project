# This code is used to generate the pruned structure
from transformers import AutoModelForMaskedLM
import torch
# roberta = AutoModelForMaskedLM.from_pretrained("roberta-base")
# roberta.config.intermediate_size = int(3072*0.02)
# newroberta = AutoModelForMaskedLM.from_config(roberta.config)
# for k in range(3):
#     newroberta.roberta.encoder.layer[k].intermediate = roberta.roberta.encoder.layer[k].intermediate
#     newroberta.roberta.encoder.layer[k].output.dense = roberta.roberta.encoder.layer[k].output.dense
# torch.save(newroberta,"prune_structure/PruneRoberta")
# print(sum(x.numel() for x in newroberta.parameters())/sum(x.numel() for x in roberta.parameters()))

bert = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
bert.config.intermediate_size = int(3072*0.02)
newbert = AutoModelForMaskedLM.from_config(bert.config)
for k in range(3):
    newbert.bert.encoder.layer[k].intermediate = bert.bert.encoder.layer[k].intermediate
    newbert.bert.encoder.layer[k].output.dense = bert.bert.encoder.layer[k].output.dense
torch.save(newbert,"prune_structure/PruneBert")
print(sum(x.numel() for x in newbert.parameters())/sum(x.numel() for x in bert.parameters()))