import math
import random
import logging
import torch

# prompts
_PRIMARY_PROMPT = (
    """{}\nQuestion: Does this imply that {}?\nOPTIONS:\nEntailment\nContradiction"""
)
_SUMMARY_PROMPT = "Please summarize the following and include important details as much as possible: {}"
_PREDICT_SECTIONID_PROMPT = "A premise contains four sections: {}\nA hypothesis describes one of the sections: {}\nDetermine the most relevant section from the four options: intervention, results, eligibility, adverse_events"

# word2token
_WORD_TO_TOKEN_RATIO = 4
_MODEL_TOKEN_LIMIT = 512
# sliding window
_STRIDE = 256
_SUMMARY_GEN_MAX_TOKEN = 240
_SECTION_ORDER = ["intervention", "eligibility", "adverse_events", "results"]

def ans_to_label(result):
    if result == "yes":
        return "entailment"
    elif result == "no":
        return "contradiction"
    return result


class EntModel:
    def __init__(
        self,
        entailment_prompt=_PRIMARY_PROMPT,
        summary_prompt=_SUMMARY_PROMPT,
        predict_section_prompt=_PREDICT_SECTIONID_PROMPT,
        w2t_ratio=_WORD_TO_TOKEN_RATIO,
        model_token_limit=_MODEL_TOKEN_LIMIT,
        stride=_STRIDE,
        summary_max_output_token=_SUMMARY_GEN_MAX_TOKEN,
        device='cpu'
    ) -> None:
        self.model = None
        self.tokenizer = None
        self.device = device

        # TODO
        self.summary_model = None
        self.summary_tokenizer = None

        self.prompt_template = entailment_prompt
        self.summary_template = summary_prompt
        self.predict_section_template = predict_section_prompt
        self.w2t_ratio = w2t_ratio
        self.model_token_limit = model_token_limit
        self.stride = stride
        self.summary_max_output_token = summary_max_output_token

    def setModels(self, model, tokenizer):
        if not model or not tokenizer:
            raise ValueError("must set both model and tokenizer")

        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    def get_prompt(self, premise, hypothesis):
        return self.prompt_template.format(premise, hypothesis)

    # LLM QA method
    def chat_entailment(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=20)
        return ans_to_label(
            self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].lower()
        )

    # truncate premise by model_max_length
    def truncate_premise(self, premise, hypothesis):
        max_token = (self.model_token_limit - 
                     (len(hypothesis) + len(self.prompt_template)) // self.w2t_ratio)
        
        # require min of 10 token or 40 characters to make up the premise.
        # return "" to assing this example an output "unknown"
        if max_token <= 10:
            return ""
        tokens = self.tokenizer.encode(premise, truncation=True, max_length=max_token)
        truncated = self.tokenizer.decode(tokens, skip_special_tokens=True)

        debug_original = self.tokenizer.decode(self.tokenizer.encode(premise), skip_special_tokens=True)
        logging.info(f'''model_token_limit: {self.model_token_limit}, max_token: {max_token}
        premise: {len(premise)}, unchanged: {len(debug_original)}, after: {len(truncated)}
        ''')
        return truncated

    # LLM QA method
    def chat_summary(self, prompt, output_max_token=_SUMMARY_GEN_MAX_TOKEN):
        tokenizer = self.summary_tokenizer if self.summary_tokenizer else self.tokenizer
        model = self.summary_model if self.summary_model else self.model
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=output_max_token)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    # helper functions
    def predict_section(self, premise, hypothesis):
        """predict seciton id by premise, hypothesis
            return: str
        """
        prompt = self.predict_section_template.format(premise, hypothesis)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip().lower()

        if pred == 'interventions':
            pred = 'intervention'
        if pred not in ["intervention", "eligibility", "adverse_events", "results"]:
            pred = "unknown"
        return pred

    def generate_summary(self, long_text, concat=True, output_max_token=None):
        """generate summary for input premise
        return: summary (str), tmp: List[str]
        """
        chunks = []  # summary chunks
        total_length = len(long_text)
        chunk_size = math.floor(self.model_token_limit * self.w2t_ratio * 0.9)
        start = 0

        if not output_max_token:
            output_max_token = self.summary_max_output_token

        while start < total_length:
            end = min(start + chunk_size, total_length)
            _prompt = self.summary_template.format(long_text[start:end])
            chunk_summary = self.chat_summary(
                _prompt, output_max_token=output_max_token
            )
            chunks.append(chunk_summary)
            start += chunk_size

        if concat:
            summary = ". ".join(chunks)
        elif len(chunks) > 1:
            _prompt = self.summary_template.format(". ".join(chunks))
            summary = self.chat_summary(_prompt, output_max_token=output_max_token)
        else:
            summary = chunks[0]

        return summary, chunks

    def get_windowed_premise(self, premise, hypothesis):
        """generate premise chunks with sliding window method
        make sure stride is large enough so we don't run too many sections per sample
        256 is probably the smallest you should go. max is 512. *4 for word counts.
        """

        # unit: characters
        window_size = (
            self.w2t_ratio * self.model_token_limit
            - len(hypothesis)
            - len(self.prompt_template)
            - 4
        )
        chunks = []
        total_length = len(premise)
        start = 0
        stride = self.w2t_ratio * self.stride # token -> character stride
        while start < total_length:
            end = min(start + window_size, total_length)
            chunks.append(premise[start:end])
            start += stride
        return chunks
# end of class

# dataset preprocess functions
def ctr_to_full_text(ctr, include_id=False, section_order=_SECTION_ORDER):
    """extract full text from ctr
    include_id: whether to include seciton id for each section sentence list
    """
    
    logging.info(f'full text join order: {", ".join(section_order)}')
    to_join = []
    for section_id in section_order:
        text = [section_id + ":"] + ctr.get(section_id, []) if include_id else ctr.get(section_id, [])
        to_join = to_join + text

    return "\n".join(to_join)


def get_premise_hypothesis(sample, ctrs, use_section=False, include_id=True, random_section=False,
                           section_order=_SECTION_ORDER, overwrite_section=None):
    """get premise, hypothesis, label, type from a train sample
    use_section: whether to export full ctr or section only
    full_text_include_id: when exporting full ctr, whether to include section id inside presmise
    """
    sample_type = sample["type"]
    primary_ctr = ctrs[sample["primary_id"]]

    if use_section and overwrite_section:
        if overwrite_section not in _SECTION_ORDER:
            raise ValueError(f'invalid overwrite_section {overwrite_section}.')
        section_id = overwrite_section
    elif use_section and random_section:
        section_id = random.choice(['intervention', 'eligibility', 'adverse_events', 'results'])
    else:
        section_id = sample["section_id"].lower().replace(" ", "_")
    if use_section:
        primary_text = "\n".join(primary_ctr[section_id])
    else:
        primary_text = ctr_to_full_text(primary_ctr, include_id, section_order)

    if sample_type == "Comparison":
        secondary_ctr = ctrs[sample["secondary_id"]]
        if use_section:
            secondary_text = "\n".join(secondary_ctr[section_id])
        else:
            secondary_text = ctr_to_full_text(secondary_ctr, include_id, section_order)
        premise = (
            f"Primary trial evidence are {primary_text}\n and Secondary "
            + f"trial evidence are {secondary_text}."
        )
    else:
        premise = f"Primary trial evidence are {primary_text}."

    hypothesis = sample["statement"]

    # Future experiment
    # hack: if a false section id is used, the true label shoudl be "unknown"
    # if use_section and random_section and original_section_id != section_id:
    #     label = 'unknown'
    # else:
    #     label = sample["label"]

    label = sample["label"]

    return premise, hypothesis, label, sample_type


def get_premise_hypothesis_by_section(sample, ctrs):
    """get premise, hypothesis, label, type from a train sample
    use_section: whether to export full ctr or section only
    full_text_include_id: when exporting full ctr, whether to include section id inside presmise
    """

    premises = {}

    primary_ctr = ctrs[sample["primary_id"]]
    sample_type = sample["type"]
    if sample_type == "Comparison":
        secondary_ctr = ctrs[sample["secondary_id"]]
    else:
        secondary_ctr = None

    for section_id in ["intervention", "eligibility", "results", "adverse_events"]:
        section_id = section_id.lower().replace(" ", "_")
        primary_text = "\n".join(primary_ctr[section_id])
        if sample_type == "Comparison":
            secondary_text = "\n".join(secondary_ctr[section_id])
            premise = (
                f"Primary trial evidence are {primary_text}\n and Secondary "
                + f"trial evidence are {secondary_text}."
            )
        else:
            premise = f"Primary trial evidence are {primary_text}."
        premises[section_id] = premise

    hypothesis = sample["statement"]
    label = sample["label"]

    return premises, hypothesis, label, sample_type
