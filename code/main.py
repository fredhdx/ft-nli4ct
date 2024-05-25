import logging
import torch
import json
from tqdm import tqdm
import datasets
from collections import Counter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from functions import *
from topksearcher import TopKSearcher

def setUpLogging(filename=None):
    ''' Set up the logging so that all log messages are written to a file as well as printed to the console. 
    '''
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create a FileHandler to write log messages to a file
    if filename:
        file_handler = logging.FileHandler(filename, 'w')
        file_handler.setLevel(logging.INFO)  # Set the level to the desired verbosity
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
    # Add the file handler to the root logger
    logging.getLogger('').addHandler(file_handler)

def load_data():
    ''' Load the dataset and return annotations and id_to_clinical_trial_record dictionary.
        Source: https://huggingface.co/datasets/bigbio/sem_eval_2024_task_2/tree/main

        If the dataset no longer works, please follow the official starter guide to download the dataset
        and prepare them into (annotations, id_to_clinical_trial_record) pair.
        Official Link: https://github.com/ai-systems/Task-2-SemEval-2024/blob/main/starter_script.ipynb
    '''
    annotations = datasets.load_dataset(
        "bigbio/sem_eval_2024_task_2", name="sem_eval_2024_task_2_source"
    )
    raw_texts = datasets.load_dataset(
        "bigbio/sem_eval_2024_task_2", name="sem_eval_2024_task_2_ct"
    )["train"] # train contains CT

    id_to_clinical_trial_record = {}
    for instance in raw_texts:
        id_to_clinical_trial_record[instance["clinical_trial_id"]] = instance

    return annotations, id_to_clinical_trial_record

def run_eval(
    annotations,
    id_to_clinical_trial_record,
    split_name="validation",
    logfilename=None,
    use_section=False,
    include_id=True,
    premise_method="base",
    MODEL_NAME = "large",
    STRIDE = 256,
    RANDOM_SECTION = False,
    SUMMARY_GEN_MAX_TOKEN = 240,
    WORD_TO_TOKEN_RATIO = 4,
    MODEL_TOKEN_LIMIT = 512,
    PRIMARY_PROMPT = (
        """{}\nQuestion: Does this imply that {}?\nOPTIONS:\nEntailment\nContradiction"""
    ),
    SUMMARY_PROMPT = "Please summarize the following and include important details as much as possible: {}",
    TOPK=20,
    TOPK_DB=""
):
    """Primary eval/inference function. Run with your parameter choices.

    Args:
        annotations (Dataloader): annotation dataloader 
        id_to_clinical_trial_record (Dataloader): CTR dataloader 
        split_name (str, optional): train/dev/test set to use. Defaults to "validation". Test is invalid. 
        logfilename (str, optional): filename to save inference result. Defaults to None.
        use_section (bool, optional): whether to use SECTION MODE or FULL CTR MODE. Defaults to False.
        include_id (bool, optional): whether to include SECTION ID names in full ctr text. Defaults to True.
        premise_method (str, optional): premise handling method. Defaults to "base". Available: base, truncate, 
                                                sliding_window, summarize-concat, summarize, topk, autosection.
        MODEL_NAME (str, optional): Flan-T5 model size. Defaults to "large". Available: base, large, xl, xxl
        STRIDE (int, optional): stride size in tokens for Sliding Window method. Defaults to 256.
        RANDOM_SECTION (bool, optional): whether to use random seciton ID or section orders. Defaults to False.
        SUMMARY_GEN_MAX_TOKEN (int, optional): max token allowed for summary generation. Defaults to 240.
        WORD_TO_TOKEN_RATIO (int, optional): Token to Charter ratio. Defaults to 4 in English Language.
        MODEL_TOKEN_LIMIT (int, optional): Maximum Input Token Limit for some methods. Defaults to 512.
        PRIMARY_PROMPT (tuple, optional): Entailment Prompt Template. Defaults to encoded value.
        SUMMARY_PROMPT (str, optional): Summary Generation Prompt Template. Defaults to encoded value.
        TOPK (int, optional): K value for top k method. Defaults to 20.
        TOPK_DB (str, optional): Embedding database location for Top K method. Defaults to encoded value.

    Raises:
        ValueError: TODO 

    Output:
        None

    Logs:
        {logfilename}_result.log: inference results dictionary {'acc': accuracy, 'result': [[pred, true]], 
                            'logs': additional logs for some methods}
        {logfilename}.log: console logs
    """

    logging.info(f'running eval on "{split_name}"')
    logging.info(f'use_section: {use_section}, include_id: {include_id}, word2token: {WORD_TO_TOKEN_RATIO}')
    logging.info(f'model name: {MODEL_NAME}, method: {premise_method}, random section: {RANDOM_SECTION}')
    logging.info(f'sliding window - stride: {STRIDE}, max context token: {MODEL_TOKEN_LIMIT}')
    logging.info(f'summary - max gen token: {SUMMARY_GEN_MAX_TOKEN}, max context token: {MODEL_TOKEN_LIMIT}')
    logging.info(f'topk - topk: {TOPK}')
    logging.info('\n')

    # split_name can only be train or validation
    if split_name not in annotations:
        raise ValueError(f"{split_name} must be one of {annotations.keys()}")

    # allowed experiment setup check 
    if use_section and premise_method not in ["base", "sliding_window", "truncate", "autosection"]:
        raise ValueError(
            f"when using section only, premise_method must be either base, truncate, autosection, or sliding_window. Current: {premise_method}"
        )

    acc = []  # record [pred, label] pairs
    details = []  # record additional logs for some methods

    # load entail model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f'device: {device}')
    entModel = EntModel(
        entailment_prompt=PRIMARY_PROMPT,
        summary_prompt=SUMMARY_PROMPT,
        w2t_ratio=WORD_TO_TOKEN_RATIO,
        model_token_limit=MODEL_TOKEN_LIMIT,
        stride=STRIDE,
        summary_max_output_token=SUMMARY_GEN_MAX_TOKEN,
        device=device
    )

    # load Flan-T5 model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(f"google/flan-t5-{MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(f"google/flan-t5-{MODEL_NAME}")
    entModel.setModels(model, tokenizer)

    # load top k engine if applicable
    if premise_method == 'topk':
        topksearcher = TopKSearcher(TOPK)
        topksearcher.load_vector_db(db_path=TOPK_DB)
    else:
        topksearcher = None
        
    # default section order in full CTR premise
    section_order = ["intervention", "eligibility", "adverse_events", "results"]

    # for full text with random section, create random section order for the eval
    if RANDOM_SECTION and not USE_SECTION:
        random.shuffle(section_order)

    # inference on dataset
    for i, instance in enumerate(tqdm(annotations[split_name])):
        logging.info(f"eval@{i+1}")
        details.append(f"eval@{i+1}")

        # prepare premise, hypothesis, label from data
        #   - single premise when not using summarization methods
        if premise_method not in ["summarize", "summarize-concat"]:
            premise, hypothesis, label, sample_type = get_premise_hypothesis(
                instance,
                id_to_clinical_trial_record,
                use_section=use_section,
                include_id=include_id,
                section_order=section_order,
            )
        else:
        #  - multiple premises for each section when using summarization methods
            premises_by_section, hypothesis, label, sample_type = get_premise_hypothesis_by_section(
                instance, id_to_clinical_trial_record
            )

        # inference depending on method
        if premise_method == "base":
            entail_prompt = entModel.get_prompt(premise, hypothesis)
            final_result = entModel.chat_entailment(entail_prompt)
        elif premise_method == "truncate":
            trunc_premise = entModel.truncate_premise(premise, hypothesis)
            if not trunc_premise:  # fail if truncation is larger than premise length 
                final_result = "unknown"
            else:
                entail_prompt = entModel.get_prompt(trunc_premise, hypothesis)
                final_result = entModel.chat_entailment(entail_prompt)
        elif premise_method == 'autosection':  # aka, section prediction
            # 1. generate full CTR premise
            _premise, _, _, _ = get_premise_hypothesis(
                        instance, id_to_clinical_trial_record,
                        use_section=False, include_id=True)
            # 2. make a section prediction
            pred_section = entModel.predict_section(_premise, hypothesis)
            # 3. generate entailment prompt with predicted section
            if pred_section == 'unknown':
                final_result = 'unknown'
            else:
                # get section premise like in base case
                premise, _, _, _ = get_premise_hypothesis(
                                            instance,
                                            id_to_clinical_trial_record,
                                            use_section=True,
                                            include_id=include_id,
                                            overwrite_section=pred_section
                                            )
                entail_prompt = entModel.get_prompt(premise, hypothesis)
                final_result = entModel.chat_entailment(entail_prompt)
            # log section prediction result as well
            true_section = instance['section_id'].lower().replace(" ", "_")
            logging.info(f'pred section: {pred_section}, true section: {true_section}')
        elif premise_method == "sliding_window":
            # get chunked premises
            chunks = entModel.get_windowed_premise(premise, hypothesis)
            # get prediciton per chunk premise
            tmp = []
            for j, chunk in enumerate(chunks):
                entail_prompt = entModel.get_prompt(chunk, hypothesis)
                tmp_result = entModel.chat_entailment(entail_prompt)
                tmp.append(tmp_result)
            # logging.info(f"chunks: [{', '.join(tmp)}]")
            # argmax voting
            counts = Counter(tmp)
            final_result = max(counts, key=counts.get)
        elif premise_method in ["summarize", "summarize-concat"]:
            use_concat = True if premise_method == "summarize-concat" else False
            # generate section summaries for each section
            tmp = []
            for section_id, _premise in premises_by_section.items():
                section_summary, _ = entModel.generate_summary(
                    _premise, concat=use_concat
                )
                tmp.append(section_id + ": " + section_summary)

            # concatenate section summaries into CTR summary
            premise_summary = "\n".join(tmp)
            entail_prompt = entModel.get_prompt(premise_summary, hypothesis)
            final_result = entModel.chat_entailment(entail_prompt)
        elif premise_method == 'topk':
            # retreive top k most relevant sentences as premise
            premise = topksearcher.search(hypothesis)
            entail_prompt = entModel.get_prompt(premise, hypothesis)
            final_result = entModel.chat_entailment(entail_prompt)

        # record inference result
        acc.append([final_result, label.lower()])
        logging.info(f"pred: {final_result}, label: {label.lower()}")
        details.append(f"pred: {final_result}, label: {label.lower()}")
    
    # record method accuracy
    accuracy = sum([1 for x in acc if x[0] == x[1]]) / len(acc)
    logging.info(f"accuracy: {accuracy}")

    # save result to result log file
    printed = {"acc": accuracy, "result": acc, "logs": details}
    if logfilename:
        with open(logfilename, "w") as f:
            json.dump(printed, f)
        logging.info(f"log saved to: {logfilename}")

if __name__ == "__main__":

    ###### DO NOT CHANGE
    #   - Entailment Prompt Template
    PRIMARY_PROMPT = (
        """{}\nQuestion: Does this imply that {}?\nOPTIONS:\nEntailment\nContradiction"""
    )
    #   - Summary Prompt Template
    SUMMARY_PROMPT = "Please summarize the following and include important details as much as possible: {}"
    #   - Token to Charter Ratio, 4 for English
    WORD_TO_TOKEN_RATIO = 4 
    #   - Choice of annotation set to use
    EVAL_DATA = "validation" 

    ###### DO CHANGE

    # Choose Experiment (Premise handling method)
    PREMISE_METHOD = "sliding_window"  # base, truncate, sliding_window, summarize-concat, summarize, topk, autosection

    # Choose Shared Parameters
    MODEL_NAME = "xl" # flan-t5 model size: base, large, xl, xxl
    USE_SECTION = False  # SECTION MODE (True) or FULL CTR MODE (False)
    INCLUDE_ID = True  # Include section ID as titles in full CTR mode (default True)
    # whether to choose a random section (seciton only) / randomize section orders (full text)
    RANDOM_SECTION = False  
    MODEL_TOKEN_LIMIT = 512  # maximum LLM input size limit. Set for sliding_window, truncate, summarize-concat, summarize methods. 

    # Sliding Window Parameters
    STRIDE = 1024 

    # Summarization parameters
    SUMMARY_GEN_MAX_TOKEN = 380 # unit: token

    # TOPK Parameters
    TOPK = 50
    TOPK_DB = "./vectordb/allMiniLML6V2/"  # make sure you have the vector db prepared 

    # Choose log file name
    logname = f"xl_results/sliding_1024stride_512maxtoken_xl"
    setUpLogging(f'{logname}.log')

    # load data
    annotations, id_to_clinical_trial_record = load_data()

    # run
    run_eval(
        annotations,
        id_to_clinical_trial_record,
        split_name=EVAL_DATA,
        logfilename=f'{logname}_result.log',
        use_section=USE_SECTION,
        include_id=INCLUDE_ID,
        premise_method=PREMISE_METHOD,
        MODEL_NAME=MODEL_NAME,
        STRIDE=STRIDE,
        RANDOM_SECTION=RANDOM_SECTION,
        SUMMARY_GEN_MAX_TOKEN=SUMMARY_GEN_MAX_TOKEN,
        WORD_TO_TOKEN_RATIO=WORD_TO_TOKEN_RATIO,
        MODEL_TOKEN_LIMIT=MODEL_TOKEN_LIMIT,
        PRIMARY_PROMPT=PRIMARY_PROMPT,
        SUMMARY_PROMPT=SUMMARY_PROMPT,
        TOPK=TOPK,
        TOPK_DB=TOPK_DB
    )