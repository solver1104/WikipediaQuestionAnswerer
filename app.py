import streamlit as st
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
import wikipedia
import unidecode
import math
import multitask_model
from config import *

################### SETUP ###################
st.set_page_config(
    page_title='MiniOracle',
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    with open(STS_TOPICS_PATH, 'r') as f:
        topics = eval(''.join(f.readlines()))
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
st.title("MiniOracle Demo")
question = st.text_area("Question")
TOP_K = st.slider(label="Number of Articles to Search", min_value=1, max_value=5, value=1)
search_method = st.selectbox('Query Method?', ('Closed Domain (More Accurate)', 'Open Domain (Wider Knowledge Base, SLOW)'))
with st.sidebar:
    st.title("About")
    st.header("What is MiniOracle?")
    st.caption("MiniOracle is an open domain Question Answer language model. It can accurately respond to a variety of trivia style questions (although it struggles to answer more open-ended questions without a definitive answer).")
    st.header("Errors")
    st.caption("If errors occur, try refreshing the app. Otherwise, please contact me (see below).")
    st.header("Technical Details")
    st.caption("MiniOracle comprises of a semantic search model and an extractive question answering model. The semantic search model finds Wikipedia articles that act as context for the question answering model to answer the query with. Both models are finetuned from the RoBERTa-LARGE base model. Importantly, since this app isn't just calling an API endpoint to fetch model predictions from a remote server, and instead needs to store both large models and intermediate computations produced during inferencing on the Streamlit Community Cloud servers, memory occupied by the model weights and inferencing must be judiciously reduced to fit within the 1GB RAM limits. To reduce memory usage for storing the two models (both with ~370M parameters), weight sharing is employed, dramatically reducing the storage cost. Model quantization is also used to reduce inference memory/runtime costs. For more information, see the source code repository.")
    st.link_button("Source Code", "https://github.com/solver1104/WikipediaQuestionAnswerer")
    st.divider()
    st.subheader("About the Author")
    st.write("Other Projects: [GitHub](https://github.com/solver1104)")
    st.write("Contact me: [Email](mailto:s1104@uw.edu)")
    

################### QUERY ###################
if st.button('Submit Query'):
    if len(question) != 0:
        question = unidecode.unidecode(question.strip())
        question_answered = False
        
        st.toast("Starting query", icon="💡")
        question_tokenized = tokenizer(question + tokenizer.sep_token)
        context_vecs = []

        MODULE_NUM = 1
        model.apply(switch_modules)
        question_tokenized_no_sep = tokenizer(question, padding="max_length", max_length=256, truncation=True)
        A_ids = torch.tensor(question_tokenized_no_sep["input_ids"], device=device).unsqueeze(dim=0)
        A_masks = torch.tensor(question_tokenized_no_sep["attention_mask"], device=device).unsqueeze(dim=0)

        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                out_A = model(input_ids=A_ids, attention_mask=A_masks).last_hidden_state
                A_embeds = output_head(out_A[:, 0]).squeeze().unsqueeze(dim=0)
                question_embeds = nn.functional.normalize(A_embeds)


        if search_method == 'Closed Domain (More Accurate)':
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                # Use STS model to search for related articles
                sim = (embeds @ question_embeds.T).squeeze()
            top_articles = [topics[x] for x in torch.topk(sim, TOP_K).indices]
        else:
            # Query Wikipedia for related articles first, then use STS model to find most relevant articles
            wiki_retrieve = wikipedia.search(question, results=WIKI_SEARCH)
            wiki_query = tokenizer(wiki_retrieve, padding="max_length", max_length=16, truncation=True)
            wiki_ids = torch.tensor(wiki_query["input_ids"], device=device)
            wiki_masks = torch.tensor(wiki_query["attention_mask"], device=device)

            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    out_wiki = model(input_ids=wiki_ids, attention_mask=wiki_masks).last_hidden_state
                    wiki_embeds = output_head(out_wiki[:, 0]).squeeze().unsqueeze(dim=0)
                    wiki_query_embeds = nn.functional.normalize(wiki_embeds)

                    # Use STS model to filter results
                    sim = (wiki_query_embeds @ question_embeds.T).squeeze()
            top_articles = [wiki_retrieve[x] for x in torch.topk(sim, TOP_K).indices]
        
        # Fetch Wikipedia articles
        for title in top_articles:
            for matched in wikipedia.search(title, results=1):
                st.toast("Searching Wikipedia page: " + title, icon="✅")
                try:
                    context_vecs.append(unidecode.unidecode(wikipedia.page(matched, auto_suggest=False).content))
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
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    out = model(input_ids=ids, attention_mask=mask).last_hidden_state
                    start_preds = start_head(out).squeeze()
                    end_preds = end_head(out).squeeze()
                    is_answerable_preds = is_answerable_head(out[:, 0]).squeeze()

            start_preds = torch.softmax(start_preds, dim=0)
            end_preds = torch.softmax(end_preds, dim=0)
            ids = ids.squeeze()

            if is_answerable_preds.item() <= 0 and torch.max(start_preds).item() > CONF_THRESHOLD and torch.max(end_preds).item() > CONF_THRESHOLD:
                st.success("Prediction: " + tokenizer.decode(ids[torch.argmax(start_preds).item() : torch.argmax(end_preds).item() + 1]) + ", Confidence: " + str(round(100 * min(torch.max(start_preds).item(), torch.max(end_preds).item()), 2)) + "%")
                question_answered = True
                break

        if question_answered:
            st.toast("Inference Complete", icon="💡")
        else:
            st.toast("Couldn't find answer", icon="❌")
    else:
        e = RuntimeError('Type a question before submitting a query!')
        st.exception(e)

