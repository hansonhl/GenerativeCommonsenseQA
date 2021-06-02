import math
import os
import pickle
import subprocess

import networkx as nx
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

merged_relations = [
    'antonym',
    'atlocation',
    'capableof',
    'causes',
    'createdby',
    'isa',
    'desires',
    'hassubevent',
    'partof',
    'hascontext',
    'hasproperty',
    'madeof',
    'notcapableof',
    'notdesires',
    'receivesaction',
    'relatedto',
    'usedfor',
]



def num_combinations(n, up_to=4):
    res = 0
    f = math.factorial
    for r in range(1, min(n, up_to)):
        res += f(n) // f(r) // f(n - r)
    return res + 1


def wccount(filename):
    out = subprocess.Popen(['wc', '-l', filename],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT
                           ).communicate()[0]
    return int(out.partition(b' ')[0])

def load_graph_info1(data_dir):
    # Graph info used in QAGNN
    logger.info("loading cpnet (1)....")
    with open(os.path.join(data_dir, "concept.txt"), "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}

    cpnet = nx.read_gpickle(os.path.join(data_dir, "conceptnet.en.pruned.graph"))
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)
    return {"kg_full": cpnet, "kg_simple": cpnet_simple,
            "i2r": id2relation, "r2i": relation2id,
            "i2e": id2concept, "e2i": concept2id}


def load_graph_info2(data_dir):
    # Graph info used in connect-the-dots paper
    logger.info("loading cpnet (2)....")
    data_path = os.path.join(data_dir, 'conceptnet_graph.nx')
    kg_full = nx.read_gpickle(data_path)

    kg_simple = nx.DiGraph()
    for u, v, data in kg_full.edges(data=True):
        kg_simple.add_edge(u, v)

    rel_path = os.path.join(data_dir, 'relation_vocab.pkl')
    ent_path = os.path.join(data_dir, 'entity_vocab.pkl')

    with open(rel_path, 'rb') as handle:
        rel_vocab = pickle.load(handle)

    with open(ent_path, 'rb') as handle:
        ent_vocab = pickle.load(handle)

    return {"kg_full": kg_full, "kg_simple": kg_simple,
            "i2r": rel_vocab["i2r"], "r2i": rel_vocab["r2i"],
            "i2e": ent_vocab['i2e'], "e2i": ent_vocab['e2i']}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def batched_decoding_loss(loss_fxn, logits, labels, weights=None, pad_token_id=1):
    batch_size, batch_max_len, vocab_size = logits.shape
    labels = labels.reshape(-1)
    logits = logits.reshape(-1, vocab_size)
    loss = loss_fxn(logits, labels)
    mask = labels != pad_token_id
    loss[~mask] = 0.
    loss = loss.reshape(batch_size, -1).sum(dim=1)
    lengths = mask.reshape(batch_size, -1).sum(dim=1)
    loss = loss / lengths
    if weights is not None:
        loss = loss * weights
    return torch.mean(loss)

def get_gpt2_position_ids(q_input_ids, pos_id_offsets, a_input_ids, include_q=True):
    batch_size = pos_id_offsets.shape[0]

    max_q_len = q_input_ids.shape[1]
    max_pos_id_len = pos_id_offsets.shape[1]

    question_pos_ids = torch.arange(max_q_len).repeat(batch_size, 1)
    sep_pos_ids1 = torch.full((batch_size, 1), max_q_len)
    concept_pos_ids = pos_id_offsets + max_q_len + 1

    sep_pos_ids2 = torch.full((batch_size, 1), max_q_len + 1 + max_pos_id_len, dtype=torch.long)

    if a_input_ids is None:
        if include_q:
            return torch.cat((question_pos_ids, sep_pos_ids1, concept_pos_ids, sep_pos_ids2), dim=1)
        else:
            return torch.cat((sep_pos_ids1, concept_pos_ids, sep_pos_ids2), dim=1)
    else:
        max_answer_len = a_input_ids.shape[1]
        answer_pos_ids = torch.arange(
            max_q_len + 1 + max_pos_id_len + 1,
            max_q_len + 1 + max_pos_id_len + 1 + max_answer_len
        ).repeat(batch_size, 1)
        if include_q:
            return torch.cat((question_pos_ids, sep_pos_ids1, concept_pos_ids, sep_pos_ids2, answer_pos_ids), dim=1)
        else:
            return torch.cat((sep_pos_ids1, concept_pos_ids, sep_pos_ids2, answer_pos_ids), dim=1)

