import streamlit as st
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
import wikipedia
import unidecode
import math
import multitask_model
from config import *

################### LOADING ###################

@st.cache_resource
def load_model():
    return multitask_model.setup_QA()

@st.cache_resource
def load_tokenizer():
    return RobertaTokenizer.from_pretrained("roberta-large")

@st.cache_resource
def load_embeds():
    return torch.load(STS_EMBEDS_PATH, map_location=device)

@st.cache_resource
def load_topics():
    topics = []
    
    # Get all "Vital Article" titles
    article=wikipedia.page('Wikipedia:Vital articles/List of all articles')
    topics = article.content.replace('\n\n',' · ').split(" · ")[1:]

    # Remove accented characters to make the embedding process better
    for i in range(len(topics)):
        topics[i] = unidecode.unidecode(topics[i])
    return topics

with st.spinner('Loading model components...'):
    topics = load_topics()
    print("Topics loaded")
    
    embeds = load_embeds()
    print("Embeddings loaded")
    
    tokenizer = load_tokenizer()
    print("Tokenizer loaded")

    model, start_head, end_head, is_answerable_head, output_head = load_model()
    model.eval()
    start_head.eval()
    end_head.eval()
    is_answerable_head.eval()
    output_head.eval()
    print("Models loaded")
    print("Ready for inference")

################### MODEL UTILS ###################

MODULE_NUM = 0 # Task number to perform

def switch_modules(m):
    if type(m) is multitask_model.LoRA or type(m) is multitask_model.InterOutputAdapter:
        m.selected_module = MODULE_NUM

################### WEBAPP ###################
st.title("Open Domain Question Answerer")
question = st.text_area("Query the model!").strip()
TOP_K = st.slider(label="Query top n Wikipedia articles matching query", min_value=1, max_value=5, value=1)

if st.button('Run Query'):
    if len(question) != 0:
        question_answered = False
        
        st.toast("Starting query", icon="💡")
        question_tokenized = tokenizer(question + tokenizer.sep_token)
        context_vecs = []

        # Use STS model to search for related articles
        MODULE_NUM = 1
        model.apply(switch_modules)
        
        question_tokenized_no_sep = tokenizer(question, padding="max_length", max_length=256, truncation=True)

        A_ids = torch.tensor(question_tokenized_no_sep["input_ids"], device=device).unsqueeze(dim=0)
        A_masks = torch.tensor(question_tokenized_no_sep["attention_mask"], device=device).unsqueeze(dim=0)

        with torch.no_grad():
            out_A = model(input_ids=A_ids, attention_mask=A_masks).last_hidden_state
            A_embeds = output_head(out_A[:, 0]).squeeze().unsqueeze(dim=0)
            A_norm = nn.functional.normalize(A_embeds)

        sim = (embeds @ A_norm.T).squeeze()
        top_articles = [topics[x] for x in torch.topk(sim, TOP_K).indices]

        # Fetch Wikipedia articles
        for title in top_articles:
            for matched in wikipedia.search(title, results=1):
                st.toast("Searching Wikipedia page: " + title, icon="✅")
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

        # Query model
        MODULE_NUM = 0
        model.apply(switch_modules)
        
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
                question_answered = True
                break

        if question_answered:
            st.toast("Inference Complete", icon="💡")
        else:
            st.toast("Couldn't find answer", icon="❌")
    else:
        e = RuntimeError('Type a question before submitting a query!')
        st.exception(e)
