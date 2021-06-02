import os
import json
import math
import pickle
from argparse import Namespace

from scipy.special import binom
from itertools import combinations

import torch
from torch.utils.data import Dataset
import numpy as np

from tqdm import tqdm
import transformers

from typing import *

from utils import num_combinations, load_graph_info1, load_graph_info2, get_gpt2_position_ids

MAX_QUESTION_LEN = 31
MAX_NUM_GOLD_TRIPLES = 600

valid_model_types = ["bart", "gpt2"]

def validate_tokenizer_type(tokenizer, model_type):
    if model_type == "gpt2" and not isinstance(tokenizer, transformers.GPT2Tokenizer) \
        or model_type == "bart" and not isinstance(tokenizer, transformers.BartTokenizer):
        raise ValueError(f"Tokenizer {tokenizer} and model type {model_type} do not match!")


def copy_args(args):
    return Namespace(**vars(args))

def multi_random_samples(s, n: int, m: int) -> List[List]:
    # get m random samples of size n from sequence of items s
    if not isinstance(s, list):
        s = list(s)
    if len(s) <= n:
        return [s]
    s = np.array(s)
    possible_combinations = binom(len(s), n)
    res = []
    if possible_combinations > 2 * m:
        encountered = set()
        while len(encountered) < m:
            idxs = tuple(sorted(np.random.permutation(len(s))[:n]))
            if idxs not in encountered:
                encountered.add(idxs)
        for idxs in encountered:
            sel_idxs = np.array(idxs)
            res.append(s[sel_idxs].tolist())
    else:
        idxs = np.random.permutation(len(s))
        for i, sel_idxs in enumerate(combinations(idxs, n)):
            if i >= m: break
            sel_idxs = np.array(sel_idxs)
            res.append(s[sel_idxs].tolist())

    return res




def np_random_sample(s, n: int) -> List:
    if not isinstance(s, list):
        s = list(s)
    s = np.array(s)
    idxs = np.random.permutation(len(s))[:min(n, len(s))]
    return s[idxs].tolist()

def np_random_permutation(s) -> List:
    if not isinstance(s, list):
        s = list(s)
    s = np.array(s)
    shuffle_idxs = np.random.permutation(len(s))
    return s[shuffle_idxs].tolist()


class AnsgenTrainingDatahelper:
    """docstring for DataHelper"""
    def __init__(
            self, tokenizer, args, teacher_forcing=True
    ):
        if args.model_type not in valid_model_types:
            raise RuntimeError(f"Invalid model type {args.model_type}")
        validate_tokenizer_type(tokenizer, args.model_type)

        self.tokenizer = tokenizer
        if "cpnet2" in args.graph_info_dir:
            self.graph_info = load_graph_info2(args.graph_info_dir)
        else:
            self.graph_info = load_graph_info1(args.graph_info_dir)
        self.model_type = args.model_type

        if not args.eval_only:
            train_file_name = "train.pk" if args.ans_concept_input_type == "path_concepts" else "train.grounded.jsonl"
            self.trainset = AnswerGeneratorDataset(
                tokenizer,
                os.path.join(args.preprocessed_dir, train_file_name),
                self.graph_info, args,
                teacher_forcing=True
            )
            dev_file_name = "dev.pk" if args.ans_concept_input_type == "path_concepts" else "dev.grounded.jsonl"
            self.devset = AnswerGeneratorDataset(
                tokenizer,
                os.path.join(args.preprocessed_dir, dev_file_name),
                self.graph_info,
                args,
                teacher_forcing=True
            )

        crowd_dev_file_name = "crowdsource_dev.pk" if args.ans_concept_input_type == "path_concepts" else "crowdsource_dev.grounded.jsonl"
        crowd_dev_args = copy_args(args)
        crowd_dev_args.num_noisy_examples = 1
        self.crowd_devset = AnswerGeneratorDataset(
            tokenizer,
            os.path.join(args.preprocessed_dir, crowd_dev_file_name),
            self.graph_info,
            crowd_dev_args,
            teacher_forcing=False
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
    def __init__(
            self, tokenizer, grounded_examples, graph_info, args,
            teacher_forcing=True):
        """
        :param tokenizer: huggingface tokenizer that matches with the model used
        :param grounded_examples: path to .jsonl file with examples or a list of
            dicts containing questions, answers and grounded concepts
        :param graph_info: dict containing graph information
        :param ans_concept_input_type: how to present answer concepts as input to
            answer generator. Options: "combination", "all", "single"
        """
        super().__init__()
        if args.model_type not in valid_model_types:
            raise RuntimeError(f"Invalid model type {args.model_type}")
        validate_tokenizer_type(tokenizer, args.model_type)

        self.tokenizer = tokenizer
        self.prepare_special_tokens()
        self.graph_info = graph_info
        self.ans_concept_input_type = args.ans_concept_input_type
        self.max_additional_concepts = args.max_additional_concepts
        self.num_noisy_examples = args.num_noisy_examples
        self.no_permutation_invariant = args.no_permutation_invariant
        self.formatting = args.formatting
        self.model_type = args.model_type
        self.teacher_forcing = teacher_forcing

        if isinstance(grounded_examples, str):
            if self.ans_concept_input_type == "path_concepts":
                with open(grounded_examples, "rb") as f:
                    grounded_examples = pickle.load(f)
            else:
                with open(grounded_examples, "r") as f:
                    grounded_examples = [json.loads(line.strip()) for line in f]
    
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
        self.examples: List[Dict] = []
        self.qid2q = {}
        self.qid2qc = {}
        # all_concepts = set()

        for ex_dict in tqdm(grounded_examples, desc="Loading Examples"):
            qid = ex_dict["qid"]
            q = ex_dict["q"]
            if self.formatting == "possible":
                q = f"Question: {q} Possible answers: "

            question = self.tokenizer.encode(q, add_special_tokens=False)
            self.qid2q[qid] = question
            self.qid2qc[qid] = ex_dict["qc"]
            # perform padding in collate

            e2i = self.graph_info["e2i"]
            g = self.graph_info["kg_simple"]

            all_concepts = set()

            if self.ans_concept_input_type == "everything":
                # one example for one question containing all answer concepts
                for answer, gold_concepts in ex_dict["ac"].items():
                    all_concepts.update(e2i[c] for c in gold_concepts)

            for answer, gold_concepts in ex_dict["ac"].items():
                answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
                curr_concept_set = {e2i[c] for c in gold_concepts}
                if len(curr_concept_set) == 0: continue

                all_new_concepts = [set()]
                if self.max_additional_concepts > 0:
                    if self.ans_concept_input_type == "random_neighbors":
                        new_concepts = set()
                        for c in gold_concepts:
                            new_concepts.update(g.neighbors(e2i[c]))

                    elif self.ans_concept_input_type == "everything":
                        # print(f"{all_concepts=}")
                        new_concepts = all_concepts - curr_concept_set

                    elif self.ans_concept_input_type == "path_concepts":
                        paths = ex_dict["a2paths"].get(answer, [])
                        new_concepts = set()
                        for p in paths:
                            new_concepts.update(p[1:-1])
                    else:
                        raise ValueError(f"Invalid value for ans_concept_input_type "
                                         f"{self.ans_concept_input_type}")

                    if len(new_concepts) == 0: continue


                    if self.num_noisy_examples == 1:
                        all_new_concepts = [np_random_sample(new_concepts, self.max_additional_concepts)]
                    else:
                        all_new_concepts = multi_random_samples(new_concepts, self.max_additional_concepts, self.num_noisy_examples)

                for new_concepts in all_new_concepts:
                    concepts_to_add = np_random_permutation(set(new_concepts) | curr_concept_set)
                    concepts_to_add = [self.graph_info["i2e"][c] for c in concepts_to_add]
                    self._encode_and_add_example(qid, answer_ids, concepts_to_add, gold_concepts, 1.)


    def _encode_and_add_example(self, qid, answer, concepts, gold_concepts, weight):
        concept_ids = []
        position_id_offsets = []
        for i, c in enumerate(concepts):
            if self.no_permutation_invariant:
                c += ", "
            ids = self.tokenizer.encode(c.replace("_", " "), add_special_tokens=False)
            concept_ids.extend(ids)
            if not self.no_permutation_invariant:
                position_id_offsets.extend(list(range(len(ids))))

        if self.no_permutation_invariant:
            position_id_offsets = list(range(len(concept_ids)))

        self.examples.append({"qid": qid, "answer": answer,
                              "ac": concept_ids, "weight": weight,
                              "position_id_offset": position_id_offsets,
                              "ac_str": concepts, "gold_acs": gold_concepts})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        res = self.examples[item].copy()
        qid = res["qid"]

        question = self.qid2q[qid]
        res["question"] = question

        if self.model_type == "bart":
            res["context"] = question + res["ac"]
        return res

    def collate_fn(self, batch):
        if self.model_type == "bart":
            return bart_collate_fn(self.tokenizer, batch)
        elif self.model_type == "gpt2":
            return gpt2_collate_fn(self.tokenizer, self.teacher_forcing, batch, formatting=self.formatting)
        else:
            raise RuntimeError(f"Invalid model type {self.model_type}")


def bart_collate_fn(tokenizer, batch):
    ctx_lens = torch.tensor([len(ex_dict["context"]) for ex_dict in batch])
    sort_idxs = torch.argsort(ctx_lens, descending=True)
    qids = [batch[i]["qid"] for i in sort_idxs]
    ac_strs = [batch[i]["ac_str"] for i in sort_idxs]
    weights = None
    if all("weight" in d for d in batch):
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

def gpt2_collate_fn(tokenizer, teacher_forcing, batch, formatting="SEP"):
    # q_lens = torch.tensor([len(ex_dict["question"]) for ex_dict in batch])
    # idxs = torch.argsort(q_lens, descending=True)
    idxs = list(range(len(batch)))

    qids = [batch[i]["qid"] for i in idxs]
    question_ids = [torch.tensor(batch[i]["question"]) for i in idxs]
    question_lens = torch.tensor([len(q) for q in question_ids])
    padded_question_ids = torch.nn.utils.rnn.pad_sequence(
        question_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    sep_tokens = torch.tensor([tokenizer.sep_token_id] * len(batch)).unsqueeze(1)
    answer_ids = [torch.tensor(batch[i]["answer"] + [tokenizer.eos_token_id]) for i in idxs]
    padded_answer_ids = torch.nn.utils.rnn.pad_sequence(
        answer_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    max_answer_len = padded_answer_ids.shape[1] + 1
    if formatting == 'possible':
        answer_hint = [torch.tensor(tokenizer.encode("Answer:")) for _ in idxs]
        packed_answer_hints = torch.nn.utils.rnn.pad_sequence(answer_hint, batch_first=True,
                                                              padding_value=tokenizer.pad_token_id)
        padded_answer_ids = torch.cat((packed_answer_hints, padded_answer_ids), dim=1)
        if not teacher_forcing:
            padded_answer_ids = packed_answer_hints
    elif not teacher_forcing:
        padded_answer_ids = None

    if any(batch[i]["ac"] is None for i in idxs):
        input_ids = torch.cat((padded_question_ids, sep_tokens, padded_answer_ids), dim=1)
        ctx_attention_mask = torch.where(input_ids != tokenizer.pad_token_id, 1., 0.)
        return {"qids": qids, "input_ids": input_ids,
            "answer_ids": padded_answer_ids if not teacher_forcing else None,
            "attention_mask": ctx_attention_mask,
            "max_answer_len": max_answer_len}

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
    position_ids = get_gpt2_position_ids(
        padded_question_ids, question_lens, padded_position_id_offsets,
        padded_answer_ids, formatting=formatting)

    if formatting == "SEP":
        if padded_answer_ids is not None:
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
    else:
        if padded_answer_ids is not None:
            input_ids = torch.cat((padded_question_ids,
                                   padded_concept_ids,
                                   padded_answer_ids), dim=1)
        else:
            input_ids = torch.cat((padded_question_ids,
                                   padded_concept_ids,), dim=1)

    ctx_attention_mask = torch.where(input_ids != tokenizer.pad_token_id, 1., 0.)

    return {"qids": qids, "input_ids": input_ids,
            "answer_ids": padded_answer_ids if not teacher_forcing else None,
            "attention_mask": ctx_attention_mask,
            "position_ids": position_ids,
            "weights": weights, "ac_strs": ac_strs,
            "max_answer_len": None if not teacher_forcing else max_answer_len}

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




