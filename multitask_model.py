import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
import math
from config import *

class LoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.lin_weight = linear.weight
        self.lin_bias = linear.bias
        self.selected_module = 0

        # LoRA parameters (one set per task)
        self.LoRA_A = []
        self.LoRA_B = []
    def forward(self, inp):
        assert self.selected_module < len(self.LoRA_A), f"Weight module {self.selected_module} does not exist"
        return inp @ (self.lin_weight + self.LoRA_A[self.selected_module] @ self.LoRA_B[self.selected_module]).T + self.lin_bias
    # Load parameters from array containing LoRA_A and LoRA_B
    def load_params(self, params):
        self.LoRA_A.append(params[0])
        self.LoRA_B.append(params[1])

# Identity function
class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return inp

# Replacing HuggingFace Transformer's RoBERTa Intermediate and Output layers, so we can use the parallel adapter scheme
# For more details, see the source code for RoBERTa here: https://huggingface.co/transformers/v3.2.0/_modules/transformers/modeling_roberta.html
class InterOutputAdapter(torch.nn.Module):
    def __init__(self, inter, output, inp_sz, out_sz, bottleneck):
        super().__init__()
        self.selected_module = 0
        self.inter = inter
        self.output = nn.Sequential(
            output.dense,
            output.dropout
        )
        self.out_layernorm = output.LayerNorm
        self.adapter = []
        self.inp_sz = inp_sz
        self.out_sz = out_sz
        self.bottleneck = bottleneck

    # inter_inp will always be a copy of inp here due to the way we're setting up the adapter
    def forward(self, inp, inter_inp):
        assert self.selected_module < len(self.adapter), f"Weight module {mod_num} does not exist"
        adapter_output = self.adapter[self.selected_module](inp)
        main_net = self.output(self.inter(inp))

        return self.out_layernorm(main_net + inp + adapter_output)

    # Load adapter parameters
    def load_params(self, params):
        temp_adapter = nn.Sequential(
            nn.Linear(self.inp_sz, self.bottleneck),
            nn.Tanh(),
            nn.Linear(self.bottleneck, self.out_sz)
        )
        
        temp_adapter[0].weight = params[0]
        temp_adapter[0].bias = params[1]
        temp_adapter[2].weight = params[2]
        temp_adapter[2].bias = params[3]
        self.adapter.append(temp_adapter)

def setup_QA():
    # Load model from HuggingFace
    model = RobertaModel.from_pretrained("roberta-large")
    
    # Add LoRA and Adapter parameters to model

    for i in range(len(model.encoder.layer)):
        model.encoder.layer[i].attention.self.query = LoRA(model.encoder.layer[i].attention.self.query, RANK, ALPHA)
        model.encoder.layer[i].attention.self.key = LoRA(model.encoder.layer[i].attention.self.key, RANK, ALPHA)
        model.encoder.layer[i].attention.self.value = LoRA(model.encoder.layer[i].attention.self.value, RANK, ALPHA)
        model.encoder.layer[i].attention.output.dense = LoRA(model.encoder.layer[i].attention.output.dense, RANK, ALPHA)

        model.encoder.layer[i].output = InterOutputAdapter(model.encoder.layer[i].intermediate, model.encoder.layer[i].output, EMB_SZ, EMB_SZ, ADAPTER_RANK)
        
        model.encoder.layer[i].intermediate = Identity()
    
    start_head = nn.Linear(EMB_SZ, 1, bias=HEAD_BIAS)
    end_head = nn.Linear(EMB_SZ, 1, bias=HEAD_BIAS)
    is_answerable_head = nn.Linear(EMB_SZ, 1, bias=HEAD_BIAS)
    output_head = nn.Linear(EMB_SZ, OUT_EMB_SZ, bias=HEAD_BIAS)

    # Load fine-tuned parameters into model
    x = torch.load(QA_LoRA_PATH, map_location=device)
    y = torch.load(STS_LoRA_PATH, map_location=device)

    cnt = 0

    for i in range(len(model.encoder.layer)):
        model.encoder.layer[i].attention.self.query.load_params(x["LoRA" + str(cnt)])
        model.encoder.layer[i].attention.self.query.load_params(y["LoRA" + str(cnt)])
        cnt += 1

        model.encoder.layer[i].attention.self.key.load_params(x["LoRA" + str(cnt)])
        model.encoder.layer[i].attention.self.key.load_params(y["LoRA" + str(cnt)])
        cnt += 1

        model.encoder.layer[i].attention.self.value.load_params(x["LoRA" + str(cnt)])
        model.encoder.layer[i].attention.self.value.load_params(y["LoRA" + str(cnt)])
        cnt += 1

        model.encoder.layer[i].attention.output.dense.load_params(x["LoRA" + str(cnt)])
        model.encoder.layer[i].attention.output.dense.load_params(y["LoRA" + str(cnt)])
        cnt += 1

    x = torch.load(QA_ADAPTERS_PATH, map_location=device)
    y = torch.load(STS_ADAPTERS_PATH, map_location=device)

    cnt = 0
    for i in range(len(model.encoder.layer)):
        model.encoder.layer[i].output.load_params(x["adapter" + str(cnt)])
        model.encoder.layer[i].output.load_params(y["adapter" + str(cnt)])
        cnt += 1

    start_head.load_state_dict(torch.load(QA_START_HEAD_PATH, map_location=device))
    end_head.load_state_dict(torch.load(QA_END_HEAD_PATH, map_location=device))
    is_answerable_head.load_state_dict(torch.load(QA_IS_ANSWERABLE_HEAD_PATH, map_location=device))
    output_head.load_state_dict(torch.load(STS_OUTPUT_HEAD_PATH, map_location=device))
    
    return model, start_head, end_head, is_answerable_head, output_head
