import streamlit as st
import torch
from torch import nn as nn
from transformers import RobertaModel, RobertaTokenizer
import wikipedia
import math

################### MODEL ###################

config = {
    "RANK": 16,
    "ALPHA": 16,
    "EPOCHS": 2,
    "BATCH_SZ": 8, # Tune batch size depending on how much RAM your GPU has
    "GRAD_ACCU": 3,
    "LoRA_LR": 1e-4,
    "HEAD_LR": 1e-4,
    "LoRA_WD": 0.01,
    "HEAD_WD": 0.01,
    "HEAD_BIAS": True,
    "ADAPTER_WEIGHT_INIT": 0.005,
    "ADAPTER_RANK": 256,
    "ADAPTER_WD": 0.01,
    "ADAPTER_LR": 1e-4,
    "WARMUP_RATIO": 0.1
}

RANK = config["RANK"]
ALPHA = config["ALPHA"]
EPOCHS = config["EPOCHS"]
BATCH_SZ = config["BATCH_SZ"]
GRAD_ACCU = config["GRAD_ACCU"]
LoRA_LR = config["LoRA_LR"]
HEAD_LR = config["HEAD_LR"]
LoRA_WD = config["LoRA_WD"]
HEAD_WD = config["HEAD_WD"]
HEAD_BIAS = config["HEAD_BIAS"]
ADAPTER_WEIGHT_INIT = config["ADAPTER_WEIGHT_INIT"]
ADAPTER_RANK = config["ADAPTER_RANK"]
ADAPTER_WD = config["ADAPTER_WD"]
ADAPTER_LR = config["ADAPTER_LR"]
WARMUP_RATIO = config["WARMUP_RATIO"]
MAX_LEN = 512 # Maximum token length for RoBERTa
EMB_SZ = 1024 # Embedding dimension for RoBERTa
SEED = 13 # Seed for DataLoader generators
device = "cpu" # Run on CPU
LOCAL = True # Is the app running on a local device (Windows file system) or not
CONF_THRESHOLD = 0.5 # Filter useless results/make the model output only very confident results

if LOCAL:
    LoRA_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\trained_LoRA.pth"
    ADAPTERS_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\trained_adapters.pth"
    END_HEAD_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\trained_end_head.pth"
    START_HEAD_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\trained_start_head.pth"
    IS_ANSWERABLE_HEAD_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\trained_is_answerable_head.pth"
else:
    LoRA_PATH = "./trained_LoRA.pth"
    ADAPTERS_PATH = "./trained_adapters.pth"
    END_HEAD_PATH = "./trained_end_head.pth"
    START_HEAD_PATH = "./trained_start_head.pth"
    IS_ANSWERABLE_HEAD_PATH = "./trained_is_answerable_head.pth"

class LoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.lin_weight = linear.weight
        self.lin_bias = linear.bias

        # Add LoRA matrices to parameters
        self.LoRA_A = nn.Parameter(torch.normal(mean=0, std=1, size=(self.lin_weight.shape[0], rank)))
        self.LoRA_B = nn.Parameter(torch.zeros((rank, self.lin_weight.shape[1])))
        self.LoRA_A.requires_grad = True
        self.LoRA_B.requires_grad = True
        # Freeze pretrained model weights
        self.lin_weight.requires_grad = False
        self.lin_bias.requires_grad = False
    def forward(self, inp):
        return inp @ (self.lin_weight + self.LoRA_A @ self.LoRA_B).T + self.lin_bias
    # Get LoRA matrices
    def get_params(self):
        return [self.LoRA_A, self.LoRA_B]
    # Load parameters from array containing LoRA_A and LoRA_B
    def load_params(self, params):
        self.LoRA_A = params[0]
        self.LoRA_B = params[1]

def adapter_weight_init(m):
    if type(m) is nn.Linear:
        nn.init.normal_(m.weight, mean = 0.0, std = ADAPTER_WEIGHT_INIT)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# Identity function
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return inp

# Replacing HuggingFace Transformer's RoBERTa Intermediate and Output layers, so we can use the parallel adapter scheme
# For more details, see the source code for RoBERTa here: https://huggingface.co/transformers/v3.2.0/_modules/transformers/modeling_roberta.html
class InterOutputAdapter(nn.Module):
    def __init__(self, inter, output, inp_sz, out_sz, bottleneck):
        super().__init__()
        self.inter = inter
        self.output = nn.Sequential(
            output.dense,
            output.dropout
        )
        self.out_layernorm = output.LayerNorm
        self.adapter = nn.Sequential(
            nn.Linear(inp_sz, bottleneck),
            nn.Tanh(),
            nn.Linear(bottleneck, out_sz)
        ).apply(adapter_weight_init)

    # inter_inp will always be a copy of inp here due to the way we're setting up the adapter
    def forward(self, inp, inter_inp):
        adapter_output = self.adapter(inp)
        main_net = self.output(self.inter(inp))

        return self.out_layernorm(main_net + inp + adapter_output)

    # Get adapter parameters
    def get_params(self):
        return [self.adapter[0].weight, self.adapter[0].bias, self.adapter[2].weight, self.adapter[2].bias]

    # Load adapter parameters
    def load_params(self, params):
        self.adapter[0].weight = params[0]
        self.adapter[0].bias = params[1]
        self.adapter[2].weight = params[2]
        self.adapter[2].bias = params[3]

@st.cache_resource
def load_model():
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

    # Load fine-tuned parameters into model
    x = torch.load(LoRA_PATH, map_location=device)

    cnt = 0

    for i in range(len(model.encoder.layer)):
        model.encoder.layer[i].attention.self.query.load_params(x["LoRA" + str(cnt)])
        cnt += 1

        model.encoder.layer[i].attention.self.key.load_params(x["LoRA" + str(cnt)])
        cnt += 1

        model.encoder.layer[i].attention.self.value.load_params(x["LoRA" + str(cnt)])
        cnt += 1

        model.encoder.layer[i].attention.output.dense.load_params(x["LoRA" + str(cnt)])
        cnt += 1

    x = torch.load(ADAPTERS_PATH, map_location=device)

    cnt = 0
    for i in range(len(model.encoder.layer)):
        model.encoder.layer[i].output.load_params(x["adapter" + str(cnt)])
        cnt += 1

    start_head.load_state_dict(torch.load(START_HEAD_PATH, map_location=device))
    end_head.load_state_dict(torch.load(END_HEAD_PATH, map_location=device))
    is_answerable_head.load_state_dict(torch.load(IS_ANSWERABLE_HEAD_PATH, map_location=device))
    
    return model, start_head, end_head, is_answerable_head

@st.cache_resource
def load_tokenizer():
    return RobertaTokenizer.from_pretrained("roberta-large")

with st.spinner('Loading model components...'):

    tokenizer = load_tokenizer()
    print("Tokenizer loaded")

    model, start_head, end_head, is_answerable_head = load_model()
    model.eval()
    start_head.eval()
    end_head.eval()
    is_answerable_head.eval()
    print("Models loaded, ready for inference")

################### WEBAPP ###################
st.title("Open Domain Question Answerer")
question = st.text_area("Query the model!").strip()
topic = st.text_area("Topic of the query?").strip()
QUERY_TOP_RESULTS = st.slider(label="Query top n Wikipedia articles matching query", min_value=1, max_value=5, value=1)


if st.button('Run Query'):
    if len(question) != 0 and len(topic) != 0:
        st.toast("Starting query", icon="💡")
        question_tokenized = tokenizer(question + tokenizer.sep_token)
        context_vecs = []

        for word in [topic]:
            for matched in wikipedia.search(word, results = QUERY_TOP_RESULTS):
                st.toast("Searching Wikipedia page: " + matched, icon="✅")
                try:
                    context_vecs.append(wikipedia.page(matched, auto_suggest=False).content)
                except:
                    continue

        tokenized_contexts = tokenizer(context_vecs)

        split_contexts = []

        for x in tokenized_contexts['input_ids']:
            x = x[1: -1]
            start = 0
            question_len = len(question_tokenized['input_ids'][:-1])

            while start < len(x):
                cur_ctx = x[start: start + MAX_LEN - 1 - question_len]
                cur_ctx.append(tokenizer(tokenizer.eos_token)['input_ids'][1])
                split_contexts.append(question_tokenized['input_ids'][:-1] + cur_ctx)
                start += MAX_LEN - 1 - question_len

        prog = st.progress(0) 
        
        for idx, context in enumerate(split_contexts):
            prog.progress((idx + 1) / len(split_contexts), text="Querying Wikipedia page")
            ids = torch.tensor(context, device=device).unsqueeze(dim = 0)
            mask = torch.ones_like(ids)

            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask).last_hidden_state
                start_preds = start_head(out).squeeze()
                end_preds = end_head(out).squeeze()
                is_answerable_preds = is_answerable_head(out[:, 0]).squeeze()

            start_preds = torch.softmax(start_preds, dim=0)
            end_preds = torch.softmax(end_preds, dim=0)
            ids = ids.squeeze()

            if is_answerable_preds.item() <= 0 and torch.max(start_preds).item() > CONF_THRESHOLD and torch.max(end_preds).item() > CONF_THRESHOLD:
                st.success("Prediction: " + tokenizer.decode(ids[torch.argmax(start_preds).item() : torch.argmax(end_preds).item() + 1]) + "        Confidence: " + str(min(torch.max(start_preds).item(), torch.max(end_preds).item())))
        st.toast("Inference Complete", icon="💡")
    else:
        e = RuntimeError('Type a question and a topic before submitting a query!')
        st.exception(e)
