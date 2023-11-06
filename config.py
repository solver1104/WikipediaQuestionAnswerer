import torch

RANK = 16
ALPHA = 16
HEAD_BIAS = True
ADAPTER_RANK = 256
MAX_LEN = 512 # Maximum token length for RoBERTa
EMB_SZ = 1024 # Embedding dimension for RoBERTa
OUT_EMB_SZ = 512
device = "cuda" if torch.cuda.is_available() else "cpu" # Check if GPU available
LOCAL = True # Is the app running on a local device (Windows file system) or not
WIKI_SEARCH = 256

if LOCAL:
    QA_LoRA_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\QA_model\\trained_LoRA.pth"
    QA_ADAPTERS_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\QA_model\\trained_adapters.pth"
    QA_END_HEAD_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\QA_model\\trained_end_head.pth"
    QA_START_HEAD_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\QA_model\\trained_start_head.pth"
    QA_IS_ANSWERABLE_HEAD_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\QA_model\\trained_is_answerable_head.pth"
    STS_LoRA_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\STS_model\\trained_LoRA.pth"
    STS_ADAPTERS_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\STS_model\\trained_adapters.pth"
    STS_OUTPUT_HEAD_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\STS_model\\trained_output_head.pth"
    STS_EMBEDS_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\STS_model\\embeds.pth"
    STS_TOPICS_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\STS_model\\topics.txt"
    CLUSTER_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\embeds\\centers.npy"
    CLUSTER_TOPICS_PATH = "C:\\Users\\bohan\\WikipediaQuestionAnswerer\\embeds\\titles"
    CLUSTER_EMBEDS_PATH = "C:\\Users\\bohan\\embeds\\shard"
else:
    QA_LoRA_PATH = "./QA_model/trained_LoRA.pth"
    QA_ADAPTERS_PATH = "./QA_model/trained_adapters.pth"
    QA_END_HEAD_PATH = "./QA_model/trained_end_head.pth"
    QA_START_HEAD_PATH = "./QA_model/trained_start_head.pth"
    QA_IS_ANSWERABLE_HEAD_PATH = "./QA_model/trained_is_answerable_head.pth"
    STS_LoRA_PATH = "./STS_model/trained_LoRA.pth"
    STS_ADAPTERS_PATH = "./STS_model/trained_adapters.pth"
    STS_OUTPUT_HEAD_PATH = "./STS_model/trained_output_head.pth"
    STS_EMBEDS_PATH = "./STS_model/embeds.pth"
    STS_TOPICS_PATH = "./STS_model/topics.txt"
    CLUSTER_PATH = "./embeds/centers.npy"
    CLUSTER_TOPICS_PATH = "./embeds/titles"
    CLUSTER_EMBEDS_PATH = "Not implemented yet, need to cache embeddings efficiently"
