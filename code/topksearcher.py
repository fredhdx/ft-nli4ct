import pickle
import torch
from pathlib import Path

class TopKSearcher:
    def __init__(self, topk=20) -> None:
        self.topk = topk
        self.raw_text_db = None
        self.annotations_db = None

    def setTopK(self, k):
        self.topk = k

    def load_vector_db(self, db_path):
        db_path = Path(db_path)

        if not db_path.exists():
            raise ValueError(f'invalid path: {str(db_path)}')

        with open(str(db_path.joinpath('raw_text_db.pickle')), 'rb') as f:
            self.raw_text_db = pickle.load(f)

        with open(str(db_path.joinpath('annotations_db_val.pickle')), 'rb') as f:
            self.annotations_db = pickle.load(f)
    
    def search(self, query_text):
        raw_text_db = self.raw_text_db

        # search for hypothesis vector
        s = [x for x in self.annotations_db if x['statement'][0] == query_text]
        if not s:
            print('invalid hypothesis.')
            return ""
        
        sample = s[0]

        sample_type = sample['type'].lower()
        primary_id = sample['primary_id']
        secondary_id = sample['secondary_id']

        # (text, embeddings)
        query = sample['statement']

        # prepare 
        if sample_type.lower() == 'single':
            # [(text, embedding), ]
            db1 = [x for sec in raw_text_db[primary_id].values() for x in sec]
            topk = TopKSearcher.fit_topk(self.topk, len(db1))
            primary_text = "\n".join(TopKSearcher.search_topk_sentences(db1, query, topk))
            premise = f"Primary trial evidence are {primary_text}."
        else:
            db1 = [x for sec in raw_text_db[primary_id].values() for x in sec]
            topk = TopKSearcher.fit_topk(self.topk // 2, len(db1))
            primary_text = "\n".join(TopKSearcher.search_topk_sentences(db1, query, topk))

            db2 = [x for sec in raw_text_db[secondary_id].values() for x in sec]
            topk = TopKSearcher.fit_topk(self.topk // 2, len(db2))
            secondary_text = "\n".join(TopKSearcher.search_topk_sentences(db2, query, topk))
            premise = (
                f"Primary trial evidence are {primary_text}\n and Secondary "
                + f"trial evidence are {secondary_text}."
            )

        return premise

    @staticmethod
    def fit_topk(topk, db_len):
        while topk > db_len:
            topk -= 1
        return topk

    @staticmethod
    def find_topk_tensors(query_tensor, tensor_list, topk):
        tensor_stack = torch.stack(tensor_list)
        similarity_scores = torch.nn.functional.cosine_similarity(query_tensor.unsqueeze(0), tensor_stack, dim=1)
        topk_indices = torch.topk(similarity_scores, k=topk).indices
        return topk_indices

    @staticmethod
    def search_topk_sentences(fulldoc, hypothesis, topk):
        topk_indicies = TopKSearcher.find_topk_tensors(hypothesis[1], [_[1] for _ in fulldoc], topk)
        found = []
        for index in topk_indicies:
            found.append(fulldoc[index.item()][0])
        return found