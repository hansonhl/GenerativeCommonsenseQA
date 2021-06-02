import os
import json
import math
import itertools
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np

from tqdm import tqdm
import transformers

from utils import num_combinations, load_graph_info1, load_graph_info2, get_gpt2_position_ids

MAX_QUESTION_LEN = 31
MAX_NUM_GOLD_TRIPLES = 600

valid_model_types = ["bart", "gpt2"]

def validate_tokenizer_type(tokenizer, model_type):
    if model_type == "gpt2" and not isinstance(tokenizer, transformers.GPT2Tokenizer) \
        or model_type == "bart" and not isinstance(tokenizer, transformers.BartTokenizer):
        raise ValueError(f"Tokenizer {tokenizer} and model type {model_type} do not match!")

class AnsgenTrainingDatahelper:
    """docstring for DataHelper"""
    def __init__(self, tokenizer, grounded_json_dir, graph_info_dir,
                 answer_concepts="combination", model_type="gpt2", teacher_forcing=True):
        if model_type not in valid_model_types:
            raise RuntimeError(f"Invalid model type {model_type}")
        validate_tokenizer_type(tokenizer, model_type)

        self.tokenizer = tokenizer
        if "cpnet2" in graph_info_dir:
            self.graph_info = load_graph_info2(graph_info_dir)
        else:
            self.graph_info = load_graph_info1(graph_info_dir)
        self.model_type = model_type
        self.trainset = AnswerGeneratorDataset(
            tokenizer,
            os.path.join(grounded_json_dir, "train.grounded.jsonl"),
            self.graph_info,
            ans_concept_input_type=answer_concepts,
            model_type=model_type,
            teacher_forcing=teacher_forcing
        )
        self.devset = AnswerGeneratorDataset(
            tokenizer,
            os.path.join(grounded_json_dir, "dev.grounded.jsonl"),
            self.graph_info,
            ans_concept_input_type=answer_concepts,
            model_type = model_type,
            teacher_forcing=teacher_forcing
        )
        # self.apply_entity_subset()

    # def apply_entity_subset(self):
    #     # only use a subset of conceptnet entities that appear in questions and answers of the dataset
    #     all_concepts = self.trainset.all_concepts.union(
    #         self.devset.all_concepts)
    #     self.sid2cid = sorted(all_concepts)
    #     self.cid2sid = {e: i for i, e in enumerate(self.sid2cid)}
    #
    #     print(f"Found {len(self.sid2cid)} unique concepts in data out of "
    #           f"{len(self.graph_info['i2e'])} in conceptnet")
    #
    #     self.trainset.sid2cid = self.sid2cid
    #     self.trainset.cid2sid = self.cid2sid
    #     self.devset.sid2cid = self.sid2cid
    #     self.devset.cid2sid = self.cid2sid

    @property
    def num_entities(self):
        return len(self.graph_info["i2e"])

class AnswerGeneratorDataset(Dataset):
    def __init__(self, tokenizer, grounded_examples,
                 graph_info, ans_concept_input_type="everything",
                 model_type="gpt2", teacher_forcing=True):
        """
        :param tokenizer: huggingface tokenizer that matches with the model used
        :param grounded_examples: path to .jsonl file with examples or a list of
            dicts containing questions, answers and grounded concepts
        :param graph_info: dict containing graph information
        :param ans_concept_input_type: how to present answer concepts as input to
            answer generator. Options: "combination", "all", "single"
        """
        super().__init__()
        if model_type not in valid_model_types:
            raise RuntimeError(f"Invalid model type {model_type}")
        validate_tokenizer_type(tokenizer, model_type)

        self.tokenizer = tokenizer
        self.prepare_special_tokens()
        self.graph_info = graph_info
        self.ans_concept_input_type = ans_concept_input_type
        self.model_type = model_type
        self.teacher_forcing = teacher_forcing

        if isinstance(grounded_examples, str):
            with open(grounded_examples, "r") as f:
                examples = [json.loads(line.strip()) for line in f]
                grounded_examples = examples
    
        self.preprocess(grounded_examples)

        # self.qid2ac = preprocess_dict["qid2ac"]
        # self.qid2qc = preprocess_dict["qid2qc"]
        # self.all_concepts = preprocess_dict["all_concepts"]
        # self.sid2cid = {}
        # self.cid2sid = {}
        # self.max_ac_len = max(len(ac) for ac in self.qid2ac.values())

    def prepare_special_tokens(self):
        self.PAD = self.tokenizer.pad_token_id
        self.SEP = self.tokenizer.sep_token_id
        self.EOS = self.tokenizer.eos_token_id

        # add special SEP token
        if self.SEP is None:
            self.tokenizer.add_tokens(['<SEP>'])
            self.SEP = self.tokenizer.convert_tokens_to_ids('<SEP>')
            self.tokenizer.sep_token_id = [self.SEP]
            self.tokenizer.sep_token = '<SEP>'

        if self.PAD is None:
            self.tokenizer.add_tokens(['<PAD>'])
            self.PAD = self.tokenizer.convert_tokens_to_ids('<PAD>')
            self.tokenizer.pad_token_id = [self.PAD]
            self.tokenizer.pad_token = '<PAD>'

        if self.EOS is None:
            self.tokenizer.add_tokens(['<EOS>'])
            self.EOS = self.tokenizer.convert_tokens_to_ids('<EOS>')
            self.tokenizer.eos_token_id = [self.EOS]
            self.tokenizer.eos_token = '<EOS>'


    def preprocess(self, grounded_examples):
        self.examples = []
        self.qid2q = {}
        self.qid2qc = {}
        # all_concepts = set()

        for ex_dict in tqdm(grounded_examples, desc="Preprocess"):
            qid = ex_dict["qid"]

            question = self.tokenizer.encode(ex_dict["q"], add_special_tokens=False)
            self.qid2q[qid] = question
            self.qid2qc[qid] = ex_dict["qc"]
            # perform padding in collate

            if self.ans_concept_input_type == "everything":
                # one example for one question containing all answer concepts
                all_concepts = []
                for answer, concepts in ex_dict["ac"].items():
                    all_concepts.extend(concepts)

                for answer in ex_dict["a"]:
                    answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)


                    # self._encode_and_add_example(qid, answer, concept_str, 1.)
            else:
                # one example for one (q, a) pair
                for answer, concepts in ex_dict["ac"].items():
                    if len(concepts) == 0: continue
                    answer = self.tokenizer.encode(answer, add_special_tokens=False)
                    if self.model_type == "bart":
                        answer = [self.EOS] + answer + [self.EOS] # decoding starts with eos in BART

                    if self.ans_concept_input_type == "combination":
                        weight = 1. / math.sqrt(num_combinations(len(concepts)))
                        self._encode_and_add_example(qid, answer, concepts, weight)

                        for num_concepts in range(1, min(4, len(concepts))):
                            for concept_subset in itertools.combinations(concepts, num_concepts):
                                self._encode_and_add_example(qid, answer, concept_subset, weight)

                    elif self.ans_concept_input_type == "all":
                        self._encode_and_add_example(qid, answer, concepts, 1.)

                    elif self.ans_concept_input_type == "single":
                        concept = [concepts[np.random.randint(len(concepts))]]
                        self._encode_and_add_example(qid, answer, concept, 1.)
                    elif self.ans_concept_input_type == "no_concepts":
                        self.examples.append({
                            "qid": qid, "answer": answer, "ac": None,
                            "weight": None, "position_id_offset": None, "ac_str": None})
                    else:
                        raise  RuntimeError(f"Invalid answer concept combination type: {self.ans_concept_input_type}")

    def _encode_and_add_example(self, qid, answer, concepts, weight):
        concept_ids = []
        position_id_offsets = []
        for c in concepts:
            ids = self.tokenizer.encode(c.replace("_", " "), add_special_tokens=False)
            concept_ids.extend(ids)
            position_id_offsets.extend(list(range(len(ids))))

        self.examples.append({"qid": qid, "answer": answer,
                              "ac": concept_ids, "weight": weight,
                              "position_id_offset": position_id_offsets,
                              "ac_str": concepts})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        res = self.examples[item].copy()
        qid = res["qid"]

        question = self.qid2q[qid]
        res["question"] = question
        # res["context"] = question + [self.SEP] + res["ac"]

        # ans_concepts = [self.cid2sid[ac] for ac in self.qid2ac[qid]]
        # if len(ans_concepts) == 0: raise RuntimeError(f"Found question {qid} has 0 concepts")
        # if self.train:
        #     answer = torch.tensor([self.cid2sid[res["answer"]]])
        #     answer_weight = 1. / math.sqrt(2 + len(self.qid2ac[qid]))  # 2 as smooting factor
        # else:
        #     answer = torch.tensor(ans_concepts + [-1] * \
        #                           (self.max_ac_len - len(ans_concepts)))
        #     answer_weight = torch.tensor([])

        return res

    def collate_fn(self, batch):
        if self.model_type == "bart":
            return bart_collate_fn(self.tokenizer, batch)
        elif self.model_type == "gpt2":
            return gpt2_collate_fn(self.tokenizer, self.teacher_forcing, batch)
        else:
            raise RuntimeError(f"Invalid model type {self.model_type}")


def bart_collate_fn(tokenizer, batch):
    ctx_lens = torch.tensor([len(ex_dict["context"]) for ex_dict in batch])
    sort_idxs = torch.argsort(ctx_lens, descending=True)
    qids = [batch[i]["qid"] for i in sort_idxs]
    ac_strs = [batch[i]["ac_str"] for i in sort_idxs]
    weights = torch.tensor([batch[i]["weight"] for i in sort_idxs])
    ctx_ids = [torch.tensor(batch[i]["context"]) for i in sort_idxs]
    answer_ids = [torch.tensor(batch[i]["answer"]) for i in sort_idxs]
    # ctx_ids = sorted([torch.tensor(ex_dict["ctx"]) for ex_dict in batch],
    #                  key=len, reverse=True)
    padded_ctx_ids = torch.nn.utils.rnn.pad_sequence(
        ctx_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_answer_ids = torch.nn.utils.rnn.pad_sequence(
        answer_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    # ctx_attention_mask = torch.where(padded_ctx_ids != self.tokenizer.pad_token_id, 1., 0.)

    return {"qids": qids, "input_ids": padded_ctx_ids, "decoder_input_ids": padded_answer_ids,
            "weights": weights, "ac_strs": ac_strs}

def gpt2_collate_fn(tokenizer, teacher_forcing, batch):
    # q_lens = torch.tensor([len(ex_dict["question"]) for ex_dict in batch])
    # idxs = torch.argsort(q_lens, descending=True)
    idxs = list(range(len(batch)))

    qids = [batch[i]["qid"] for i in idxs]
    question_ids = [torch.tensor(batch[i]["question"]) for i in idxs]
    padded_question_ids = torch.nn.utils.rnn.pad_sequence(
        question_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    sep_tokens = torch.tensor([tokenizer.sep_token_id] * len(batch)).unsqueeze(1)
    answer_ids = [torch.tensor(batch[i]["answer"] + [tokenizer.eos_token_id]) for i in idxs]
    padded_answer_ids = torch.nn.utils.rnn.pad_sequence(
        answer_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    if any(batch[i]["ac"] is None for i in idxs):
        input_ids = torch.cat((padded_question_ids, sep_tokens, padded_answer_ids), dim=1)
        ctx_attention_mask = torch.where(input_ids != tokenizer.pad_token_id, 1., 0.)
        return {"qids": qids, "input_ids": input_ids,
            "answer_ids": padded_answer_ids if not teacher_forcing else None,
            "attention_mask": ctx_attention_mask,
            "max_answer_len": padded_answer_ids.shape[1] + 2}

    ac_strs = [batch[i]["ac_str"] for i in idxs]
    weights = torch.tensor([batch[i]["weight"] for i in idxs])


    concept_ids = [torch.tensor(batch[i]["ac"]) for i in idxs]
    position_id_offsets = [torch.tensor(batch[i]["position_id_offset"]) for i in idxs]



    padded_concept_ids = torch.nn.utils.rnn.pad_sequence(
        concept_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    padded_position_id_offsets = torch.nn.utils.rnn.pad_sequence(
        position_id_offsets, batch_first=True, padding_value=0
    )
    position_ids = get_gpt2_position_ids(padded_question_ids, padded_position_id_offsets, padded_answer_ids)

    if teacher_forcing:
        input_ids = torch.cat((padded_question_ids,
                               sep_tokens,
                               padded_concept_ids,
                               sep_tokens,
                               padded_answer_ids), dim=1)
    else:
        input_ids = torch.cat((padded_question_ids,
                               sep_tokens,
                               padded_concept_ids,
                               sep_tokens), dim=1)

    ctx_attention_mask = torch.where(input_ids != tokenizer.pad_token_id, 1., 0.)

    return {"qids": qids, "input_ids": input_ids,
            "answer_ids": padded_answer_ids if not teacher_forcing else None,
            "attention_mask": ctx_attention_mask,
            "position_ids": position_ids,
            "weights": weights, "ac_strs": ac_strs,
            "max_answer_len": None if not teacher_forcing else padded_answer_ids.shape[1] + 2}

class IterativePathFinderDatahelper:
    def __init__(self, tokenizer, preprocessed_dir, graph_info_dir, max_num_edges=600,
                 add_eop=False):
        self.tokenizer = tokenizer
        self.add_eop = add_eop
        self.prepare_special_tokens() # prepare special tokens first

        if "cpnet2" in graph_info_dir:
            self.graph_info = load_graph_info2(graph_info_dir)
        else:
            self.graph_info = load_graph_info1(graph_info_dir)

        if add_eop:
            # add end of path token
            self.graph_info["i2e"].append("<EOP>")
            self.graph_info["e2i"]["<EOP>"] = len(self.graph_info["i2e"]) - 1

        # add mapping from concept id to tokenized concepts
        tokenized_concept_path = os.path.join(preprocessed_dir, "tokenized_concepts.pk")
        if not os.path.exists(tokenized_concept_path):
            i2etoks = [tokenizer.encode(c.replace("_", " "), add_special_tokens=False)
                       for c in tqdm(self.graph_info["i2e"], desc="tokenizing concepts")]
            with open(tokenized_concept_path, "wb") as f:
                pickle.dump(i2etoks, f)
        else:
            with open(tokenized_concept_path, "rb") as f:
                i2etoks = pickle.load(f)
        self.graph_info["i2etoks"] = i2etoks

        self.trainset = IterativePathFinderDataset(
            os.path.join(preprocessed_dir, "train.pk"),
            tokenizer, self.graph_info, max_num_edges=max_num_edges, add_eop=add_eop)
        self.devset = IterativePathFinderDataset(
            os.path.join(preprocessed_dir, "dev.pk"),
            tokenizer, self.graph_info, max_num_edges=max_num_edges, add_eop=add_eop)
        self.crowdsourced_devset = IterativePathFinderDataset(
            os.path.join(preprocessed_dir, "crowdsource_dev.pk"),
            tokenizer, self.graph_info, max_num_edges=max_num_edges, add_eop=add_eop)
        # self.apply_entity_subset()

    def prepare_special_tokens(self):
        self.PAD = self.tokenizer.pad_token_id
        self.SEP = self.tokenizer.sep_token_id
        self.EOS = self.tokenizer.eos_token_id
        self.cid_pad = -1

        # add special SEP token
        if self.SEP is None:
            self.tokenizer.add_tokens(['<SEP>'])
            self.SEP = self.tokenizer.convert_tokens_to_ids('<SEP>')
            self.tokenizer.sep_token_id = [self.SEP]
            self.tokenizer.sep_token = '<SEP>'

        if self.PAD is None:
            self.tokenizer.add_tokens(['<PAD>'])
            self.PAD = self.tokenizer.convert_tokens_to_ids('<PAD>')
            self.tokenizer.pad_token_id = [self.PAD]
            self.tokenizer.pad_token = '<PAD>'

        if self.EOS is None:
            self.tokenizer.add_tokens(['<EOS>'])
            self.EOS = self.tokenizer.convert_tokens_to_ids('<EOS>')
            self.tokenizer.eos_token_id = [self.EOS]
            self.tokenizer.eos_token = '<EOS>'

        if self.add_eop:
            self.tokenizer.add_tokens(['<EOP>'])
            self.EOP = self.tokenizer.convert_tokens_to_ids('<EOP>')
            self.tokenizer.eop_token_id = [self.EOP]
            self.tokenizer.eop_token = '<EOP>'

    @property
    def num_relations(self):
        return len(self.graph_info["i2r"])

    @property
    def num_entities(self):
        return len(self.graph_info["i2e"])

class IterativePathFinderDataset(Dataset):
    def __init__(self, preprocessed_path, tokenizer, graph_info, max_num_edges=600,
                 add_eop=True):
        """
        :param tokenizer: huggingface tokenizer that matches with the model used
        :param preprocessed_path: path to .jsonl file with examples or a list of
            dicts containing questions, answers and grounded concepts
        :param graph_info: dict containing graph information
        :param ans_concept_input_type: how to present answer concepts as input to
            answer generator. Options: "combination", "all", "single"
        """
        super().__init__()
        self.add_eop = add_eop
        self.cid_pad = -1
        if add_eop:
            self.eop_cid = graph_info["e2i"]["<EOP>"]

        self.tokenizer = tokenizer
        self.graph_info = graph_info
        self.max_num_edges = max_num_edges

        with open(preprocessed_path, "rb") as f:
            data_dicts = pickle.load(f)

        self.load_examples(data_dicts)

    def load_examples(self, data_dicts):
        self.examples = []
        self.qid2q = {}
        self.qid2qc = {}
        # all_concepts = set()

        for ex_dict in tqdm(data_dicts, desc="Loading examples"):
            qid = ex_dict["qid"]
            if len(ex_dict["qc_ids"]) == 0: continue

            q_input_ids = self.tokenizer.encode(ex_dict["q"], add_special_tokens=False)
            self.qid2q[qid] = q_input_ids
            self.qid2qc[qid] = ex_dict["qc_ids"]
            # perform padding in collate

            for ans_str, ac_ids in ex_dict["a2ac_ids"].items():
                if len(ac_ids) == 0: continue
                a_input_ids = self.tokenizer.encode(ans_str, add_special_tokens=False) + [self.tokenizer.eos_token_id]

                paths = ex_dict["a2paths"][ans_str]
                gold_heads = []
                gold_tails = []
                for p in paths:
                    if self.add_eop:
                        gold_heads.extend(p)
                        gold_tails.extend(p[1:])
                        gold_tails.append(self.eop_cid)
                    else:
                        gold_heads.extend(p[:-1])
                        gold_tails.extend(p[1:])

                self.examples.append({
                    "qid": qid,
                    "a_input_ids": a_input_ids,
                    "a_concept_ids": list(ac_ids),
                    "gold_heads": gold_heads,
                    "gold_tails": gold_tails
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        res = self.examples[item].copy()
        qid = res["qid"]
        res["q_input_ids"] = self.qid2q[qid]
        res["q_concept_ids"] = list(self.qid2qc[qid])

        return res

    def collate_fn(self, batch):
        return iterative_path_finder_collate_fn(batch, self.tokenizer, self.cid_pad)

def iterative_path_finder_collate_fn(batch, tokenizer, cid_pad=-1):
    # q_lens = torch.tensor([len(ex_dict["question"]) for ex_dict in batch])
    # idxs = torch.argsort(q_lens, descending=True)
    idxs = list(range(len(batch)))

    qids = [batch[i]["qid"] for i in idxs]

    q_input_id_list = [torch.tensor(batch[i]["q_input_ids"]) for i in idxs]
    q_concept_id_list = [torch.tensor(batch[i]["q_concept_ids"]) for i in idxs]
    a_input_id_list = [torch.tensor(batch[i]["a_input_ids"]) for i in idxs]
    a_concept_id_list = [torch.tensor(batch[i]["a_concept_ids"]) for i in idxs]
    gold_head_list = [torch.tensor(batch[i]["gold_heads"]) for i in idxs]
    gold_tail_list = [torch.tensor(batch[i]["gold_tails"]) for i in idxs]

    q_input_ids = torch.nn.utils.rnn.pad_sequence(
        q_input_id_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    q_concept_ids = torch.nn.utils.rnn.pad_sequence(
        q_concept_id_list, batch_first=True, padding_value=cid_pad)
    a_input_ids = torch.nn.utils.rnn.pad_sequence(
        a_input_id_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    a_concept_ids = torch.nn.utils.rnn.pad_sequence(
        a_concept_id_list, batch_first=True, padding_value=cid_pad)
    gold_heads = torch.nn.utils.rnn.pad_sequence(
        gold_head_list, batch_first=True, padding_value=cid_pad)
    gold_tails = torch.nn.utils.rnn.pad_sequence(
        gold_tail_list, batch_first=True, padding_value=cid_pad)

    return {
        "qids": qids,
        "q_input_ids": q_input_ids,
        "q_concept_ids": q_concept_ids,
        "a_input_ids": a_input_ids,
        "a_concept_ids": a_concept_ids,
        "gold_heads": gold_heads,
        "gold_tails": gold_tails,
    }




