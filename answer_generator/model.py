import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import BCELoss
import numpy as np
import networkx as nx
from torch_scatter import scatter_sum

import logging
from tqdm import tqdm
from evaluator import decode_answers, score_by_generator

from utils import batched_decoding_loss, get_gpt2_position_ids
from collections import defaultdict

logger = logging.getLogger(__name__)

class IterativePathFinder(nn.Module):
    def __init__(self, transformer_model, tokenizer, graph_info,
                 args, embed_dim=None):
        super().__init__()
        self.transformer = transformer_model
        self.transformer_embedding_layer = transformer_model.transformer.wte

        if embed_dim is None:
            embed_dim = transformer_model.config.n_embd
        self.embed_dim = embed_dim

        self.tokenizer = tokenizer
        self.graph_expansion_rough_filtering = args.graph_expansion_rough_filtering
        self.filter_concepts_by_routing_score = args.filter_concepts_by_routing_score
        self.loss_reweighting = args.loss_reweighting

        self.kg_full = graph_info["kg_full"]
        self.kg_simple = graph_info["kg_simple"]
        self.id2concept = graph_info["i2e"]
        self.concept2id = graph_info["e2i"]
        self.id2relation = graph_info["i2r"]
        self.relation2id = graph_info["r2i"]
        if "i2etoks" not in graph_info:
            self.id2concepttoks = [tokenizer.encode(c.replace("_", " "), add_special_tokens=False)
                                   for c in tqdm(self.id2concept, desc="encoding concepts")]
            logging.info(f"encoded {len(self.id2concepttoks)} concepts")
        else:
            self.id2concepttoks = graph_info["i2etoks"]

        self.tok_pad = tokenizer.pad_token_id
        self.cid_pad = -1
        self.rln_pad = len(self.relation2id)

        self.num_iterations = args.num_hops
        self.max_num_rlns_per_edge = args.max_num_rlns_per_edge
        self.loss_reweighting = args.loss_reweighting

        self.lm_head = nn.Linear(embed_dim, len(tokenizer), bias=False)
        self.concept_emb_lin = nn.Linear(embed_dim * 2, embed_dim)
        self.concept_score_lin = nn.Linear(embed_dim, embed_dim * 2)
        self.triple_score_lin = nn.Linear(embed_dim, embed_dim * 5)

        self.sigmoid = nn.Sigmoid()

        self.out_linear = nn.Linear(embed_dim, 1)

        self.dist_embd = nn.Embedding(4, embed_dim)
        self.relation_embd = nn.Embedding(len(self.relation2id) + 1, embed_dim)
        
        if self.loss_reweighting is None:
            self.concept_loss_fn = torch.nn.BCEWithLogitsLoss()
            self.edge_loss_fn = torch.nn.BCEWithLogitsLoss()
        elif self.loss_reweighting == "equal":
            self.concept_loss_fn = self.equal_reweighted_loss
            self.edge_loss_fn = self.equal_reweighted_loss
        else:
            raise NotImplementedError

        self.lm_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        self.biatt = BiAttention(input_dim=embed_dim,
                                 memory_dim=embed_dim,
                                 hid_dim=embed_dim // 4,
                                 dropout=0.1)

    def vis_cids(self, cids):
        return [
            [c for c in _cids.tolist() if c != self.cid_pad]
            for _cids in cids
        ]

    def vis_edges(self, head_cids, tail_cids, head_idxs, tail_idxs, edge_labels=None):
        adj_lists = []
        gold_adj_lists = []
        for i in range(self._batch_size):
            al = defaultdict(set)
            for hi, ti in zip(head_idxs[i], tail_idxs[i]):
                if hi == self.cid_pad: continue
                hc, tc = head_cids[i][hi].item(), tail_cids[i][ti].item()
                al[hc].add(tc)
            if edge_labels:
                gal = defaultdict(set)
                gold_his = head_idxs[i][edge_labels[i]]
                gold_tis = tail_idxs[i][edge_labels[i]]
                for ghi, gti in zip(gold_his, gold_tis):
                    if ghi == self.cid_pad: continue
                    ghc, gtc = head_cids[i][ghi].item(), tail_cids[i][gti].item()
                    gal[ghc].add(gtc)
                gold_adj_lists.append(gal)
            adj_lists.append(al)
        return adj_lists, gold_adj_lists
    
    def equal_reweighted_loss(self, logits, labels):
        num_positives = torch.sum(labels).clamp(min=1.).item()
        num_negatives = labels.shape[0] - num_positives
        pos_weight = torch.tensor([num_negatives / num_positives], dtype=torch.float, device=self._device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return loss_fn(logits, labels)

    def forward(self, q_input_ids, q_concept_ids,
                a_input_ids=None, a_concept_ids=None,
                gold_heads=None, gold_tails=None,
                teacher_forcing=True, compute_loss=True, do_decode=True,
                compute_metrics=True,
                top_n_decoding_concepts=8, top_n_routing_concepts=16,
                top_n_routing_tails=8,
                max_num_edges=1200, verbose=False,
                output_vis=False,
                **kwargs):
        if compute_loss or compute_metrics:
            assert all(x is not None for x in [a_input_ids, a_concept_ids, gold_heads, gold_tails])

        self._device = q_input_ids.device
        self._batch_size = q_input_ids.shape[0]
        self._verbose = verbose
        self._vis_dict = {
            "gold_tail_counts": [[] for _ in range(self._batch_size)],
            "retrieved_gold_tail_counts": [[] for _ in range(self._batch_size)],
            "raw_cand_tail_counts": [[] for _ in range(self._batch_size)],
            "actual_cand_tail_counts": [[] for _ in range(self._batch_size)],
            # "concept_labels": [],
            # "top_routing_cids": []
        } if output_vis else None

        # encode question
        q_attention_mask = torch.ne(q_input_ids, self.tok_pad)
        transformer_outputs = self.transformer(
            input_ids=q_input_ids,
            attention_mask=q_attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=do_decode
        )
        past_key_values = transformer_outputs.past_key_values if do_decode else None

        q_hidden_states = transformer_outputs.hidden_states[-1]
        _mask = q_attention_mask.unsqueeze(2).expand(q_hidden_states.shape).float()
        q_emb = torch.max(q_hidden_states * _mask, 1)[0]

        # initialize everything that's necessary
        all_top_decoding_cids = [[] for _ in range(self._batch_size)]
        all_top_decoding_cid_probs = [[] for _ in range(self._batch_size)]

        all_top_routing_cids = [[] for _ in range(self._batch_size)]
        all_top_routing_concept_scores = [[] for _ in range(self._batch_size)]
        all_top_routing_scores = [[] for _ in range(self._batch_size)]
        all_top_routing_cid_labels = [[] for _ in range(self._batch_size)]

        all_top_edges = [set() for _ in range(self._batch_size)]

        routing_scores = torch.zeros_like(q_concept_ids, dtype=torch.float, device=self._device) # unnormalized by path length and path count
        path_counts = torch.where(q_concept_ids != self.cid_pad, 1., 0.)
        visited_cids = [set(_q_cids.tolist()) for _q_cids in q_concept_ids]
        edge_loss, concept_loss, decoding_loss = 0., 0., 0.

        # encode heads
        head_cids = q_concept_ids
        distances = torch.zeros_like(head_cids, dtype=torch.long, device=self._device)
        head_concept_emb = self.encode_concepts(head_cids, distances, q_hidden_states, q_attention_mask)
        # head_cid_mask = torch.ne(head_cids, self.cid_pad)
        # head_concept_prob = self.sigmoid(torch.matmul(head_concept_emb, self.concept_score_lin(q_emb).unsqueeze(-1)).squeeze(-1))
        # head_concept_prob = head_concept_prob.masked_fill(~head_cid_mask, 0)

        # keep track of proportion of positive samples encountered
        positive_cids = 0
        total_cids = 0
        if compute_loss or compute_metrics:
            a_cid_set = [set(c for c in cs.tolist() if c != self.cid_pad) for cs in a_concept_ids]
            # head_concept_labels = torch.tensor([
            #     [c.item() in _a_cids and c != self.cid_pad for c in _cids]
            #     for _a_cids, _cids in zip(a_cid_set, head_cids)
            # ], device=self._device)
            # _concept_preds = head_concept_prob.view(-1)[head_cid_mask.view(-1)]
            # _concept_labels = head_concept_labels.float().view(-1)[head_cid_mask.view(-1)]
            # concept_loss += self.concept_loss_fn(_concept_preds, _concept_labels)
        total_num_rough_recovered = []
        total_num_gold_truth = []
        for hop in range(1, self.num_iterations+1):
            # logger.info(f"------ Iteration {hop} -------")
            if self.graph_expansion_rough_filtering:
                tail_cids, heads, tails, triple_labels, num_recovered, num_gold_truth = self.expand_graph_with_rough_filtering(
                    head_cids, visited_cids, q_emb, gold_heads, gold_tails,
                    teacher_forcing, max_num_edges)
                # rough_recall_rates = [r for r in rough_recall_rates if r is not None]
                # avg_rough_recall_rate = sum(rough_recall_rates) / max(1., len(rough_recall_rates))
                # print(f"Avg rough recall rate {rough_recall_rate:.2%}")
                if tail_cids is None:
                    break
                total_num_rough_recovered.append(num_recovered)
                total_num_gold_truth.append(num_gold_truth)
            else:
                tail_cids, heads, tails, triple_labels = self.expand_graph(
                    head_cids, visited_cids, gold_heads, gold_tails, teacher_forcing,
                    max_num_edges)

            if (compute_loss or compute_metrics) and self._vis_dict:
                bridge_tail_sets = []
                for i in range(self._batch_size):
                    bridge_tail_idxs = tails[i][triple_labels[i] == 1.]
                    bridge_tail_cids = tail_cids[i][bridge_tail_idxs]
                    bridge_tail_set = set(bridge_tail_cids.tolist())
                    bridge_tail_sets.append(bridge_tail_set)
                    self._vis_dict["gold_tail_counts"][i].append(len(bridge_tail_set))

            # if self._vis_dict is not None:
            #     self._vis_dict["head_cids"].append(self.vis_cids(head_cids))
            #     self._vis_dict["tail_cids"].append(self.vis_cids(tail_cids))
            #     adj_list, gold_adj_list = self.vis_edges(head_cids, tail_cids, heads, tails, triple_labels)
            #     self._vis_dict["adj_lists"].append(adj_list)
            #     self._vis_dict["gold_adj_lists"].append(gold_adj_list)

            # self._visualize_graph_expansion(hop, head_cids, tail_cids, heads, tails, gold_heads, gold_tails)
            # print(f"")
            # print(f"{tail_cids.shape=} {tails.shape=} {head_cids.shape=} {heads.shape=}")


            triple_mask = torch.ne(tails, self.cid_pad)
            tail_cid_mask = torch.ne(tail_cids, self.cid_pad)

            if torch.sum(triple_mask) == 0 or torch.sum(tail_cid_mask) == 0:
                break

            if self._vis_dict:
                for i in range(self._batch_size):
                    actual_cand_tail_count = torch.sum(tail_cid_mask[i]).item()
                    if actual_cand_tail_count > 0:
                        self._vis_dict["actual_cand_tail_counts"][i].append(actual_cand_tail_count)


            tail_distances = torch.full_like(tail_cids, hop, device=self._device)
            tail_concept_emb = self.encode_concepts(tail_cids, tail_distances, q_hidden_states, q_attention_mask)
            # (batch_size, max_num_concepts, embed_dim * 2)

            relation_emb = self.encode_relations(head_cids, tail_cids, heads, tails, gold_heads, gold_tails)
            # (batch_size, max_num_edges, embed_dim)

            triple_emb = self.encode_triples(head_concept_emb, relation_emb, tail_concept_emb, heads, tails)
            # logger.info(f"triple_emb.shape {triple_emb.shape}")

            triple_scores = torch.matmul(triple_emb, self.triple_score_lin(q_emb).unsqueeze(-1)).squeeze(-1)
            triple_scores = triple_scores.masked_fill(~triple_mask, 0)
            triple_prob = self.sigmoid(triple_scores)
            #### P(e | x): (batch_size, max_num_edges)

            assert not torch.isnan(triple_scores).any()

            tail_concept_scores = torch.matmul(tail_concept_emb, self.concept_score_lin(q_emb).unsqueeze(-1)).squeeze(-1)
            tail_concept_scores = tail_concept_scores.masked_fill(~tail_cid_mask, 0)
            #### (batch_size, max_num_tail_concepts)

            assert not torch.isnan(tail_concept_scores).any()

            # update path counts
            tail_idxs = tails.masked_fill(~triple_mask, tail_cids.shape[1]-1)
            _heads = heads.clamp(min=0) # deal with -1 idx

            path_count_per_triple = torch.gather(path_counts, 1, _heads)
            path_count_per_triple = path_count_per_triple.masked_fill(~triple_mask, 0.)
            path_counts = scatter_sum(path_count_per_triple, tail_idxs, dim=1, dim_size=tail_cids.shape[1])

            # update routing scores
            routing_scores_per_triple = torch.gather(routing_scores, 1, _heads)
            routing_scores_per_triple = routing_scores_per_triple.masked_fill(~triple_mask, 0.)
            routing_scores_per_triple = routing_scores_per_triple + triple_prob
            routing_scores = scatter_sum(routing_scores_per_triple, tail_idxs, dim=1, dim_size=tail_cids.shape[1])
            norm_routing_scores = routing_scores / path_counts.clamp(min=1) / hop
            #### (batch_size, max_num_tail_concepts)

            assert not torch.isnan(norm_routing_scores).any()

            if compute_loss or compute_metrics:
                tail_concept_labels = torch.tensor([
                    [c.item() in _a_cids and c != self.cid_pad for c in _cids]
                    for _a_cids, _cids in zip(a_cid_set, tail_cids)
                ], device=self._device, dtype=torch.bool) #### (batch_size, max_num_tail_concepts)

                # if self._vis_dict:
                #     for  i in range(self._batch_size):
                #         num_gold_tails = torch.sum(tail_concept_labels[i]).item()
                #         self._vis_dict["gold_tail_counts"][i].append(int(num_gold_tails))


            # if self._vis_dict:
            #     self._vis_dict["gold_tail_cids"].append([
            #         [c.item() for c in _cids if c != self.cid_pad and c.item() in _a_cids]
            #         for _a_cids, _cids in zip(a_cid_set, tail_cids)
            #     ])

            # record most probable concepts and edges
            curr_top_routing_cids = [] # top routing cids for this iteration
            curr_top_routing_cid_idxs = []
            for i in range(self._batch_size): #enumerate(zip(tail_cids, tail_cid_mask, tail_concept_prob, norm_routing_scores)):
                _tail_cids = tail_cids[i][tail_cid_mask[i]]
                num_cids = torch.sum(tail_cid_mask[i]).item()
                if num_cids == 0:
                    curr_top_routing_cids.append(torch.tensor([self.cid_pad], device=self._device))
                    curr_top_routing_cid_idxs.append(None)
                    continue
                _concept_scores = tail_concept_scores[i][tail_cid_mask[i]]
                _, _cid_topk_idx = torch.topk(_concept_scores, min(top_n_decoding_concepts, num_cids))
                all_top_decoding_cids[i].extend(_tail_cids[_cid_topk_idx].tolist())
                all_top_decoding_cid_probs[i].extend(_concept_scores[_cid_topk_idx].tolist())

                # also find tails with highest routing scores
                _routing_scores = norm_routing_scores[i][tail_cid_mask[i]]

                _, _routing_topk_idx = torch.topk(_routing_scores, min(top_n_routing_tails, num_cids))

                curr_top_routing_cids.append(_tail_cids[_routing_topk_idx])
                curr_top_routing_cid_idxs.append(_routing_topk_idx)

                # find corresponding edges of these tails
                # _head_cids = head_cids[i]
                # _head_cids = _head_cids[_head_cids != self.cid_pad]
                # _tail_idxs = tails[i][triple_mask[i]]
                # _head_idxs = heads[i][triple_mask[i]]
                #
                # _top_tail_idx_set = set(_routing_topk_idx.tolist())
                #
                # _top_triple_mask = torch.tensor([_idx in _top_tail_idx_set for _idx in _tail_idxs.tolist()], device=self._device)
                #
                # _top_head_cids = _head_cids[_head_idxs[_top_triple_mask]]
                # _top_tail_cids = _tail_cids[_tail_idxs[_top_triple_mask]]
                # all_top_edges[i].update((h, t) for h, t in zip(_top_tail_cids.tolist(), _top_tail_cids.tolist()))

                if self.filter_concepts_by_routing_score:
                    # _, _routing_topk_idx = torch.topk(_routing_scores, min(top_n_routing_concepts, num_cids))

                    # _concepts_for_loss_mask: bool tensor filtering out gold concepts AND topk routing concepts
                    # these concepts will be used to calculate the loss eventually
                    top_routing_concepts_mask = torch.full_like(_tail_cids, False, device=self._device, dtype=torch.bool)
                    top_routing_concepts_mask[_routing_topk_idx] = True

                    if compute_loss or compute_metrics:
                        if self._vis_dict:
                            retrieved_gold_tail_set = set(_tail_cids[_routing_topk_idx].tolist())
                            intersection = retrieved_gold_tail_set.intersection(bridge_tail_sets[i])
                            self._vis_dict["retrieved_gold_tail_counts"][i].append(len(intersection))

                        _gold_concepts_mask = tail_concept_labels[i][tail_cid_mask[i]]
                        top_routing_concepts_mask[_gold_concepts_mask] = True

                    # print(f"{top_routing_concepts_mask.shape=} {_cids.shape=}")
                    # if len(_cids) == 1:
                    #     print(f"{_cids=}")

                    all_top_routing_cids[i].extend(_tail_cids[top_routing_concepts_mask].tolist())
                    all_top_routing_scores[i].append(_routing_scores[top_routing_concepts_mask])
                    all_top_routing_concept_scores[i].append(_concept_scores[top_routing_concepts_mask])
                    if compute_loss or compute_metrics:
                        all_top_routing_cid_labels[i].append(_gold_concepts_mask[top_routing_concepts_mask])

            if compute_loss:
                # edge_loss
                _triple_logits = triple_scores.view(-1)[triple_mask.view(-1)] # (total num edges in batch,)
                _triple_labels = triple_labels.view(-1)[triple_mask.view(-1)]
                assert _triple_labels.shape == _triple_labels.shape
                edge_loss += self.edge_loss_fn(_triple_logits, _triple_labels)

                # concept_loss
                if not self.filter_concepts_by_routing_score:
                    _concept_logits = tail_concept_scores.view(-1)[tail_cid_mask.view(-1)]
                    _concept_labels = tail_concept_labels.float().view(-1)[tail_cid_mask.view(-1)]
                    assert _concept_logits.shape == _concept_labels.shape
                    positive_cids += _concept_labels.sum().item()
                    total_cids += len(_concept_labels)
                    concept_loss += self.concept_loss_fn(_concept_logits, _concept_labels)

            # if self._vis_dict:
            #     self._vis_dict["top_routing_cids"].append([cids.tolist() for cids in curr_top_routing_cids])

            # update head cids
            new_cids = []
            new_cid_idxs = []
            if teacher_forcing:
                # tails that are part of gold triples become new heads
                for _tail_idxs, _tail_cids, _triple_labels in zip(tails, tail_cids, triple_labels):
                    _new_cid_idxs = torch.tensor(list(set(_tail_idxs[_triple_labels == 1.].tolist())), device=self._device)
                    if len(_new_cid_idxs) == 0:
                        # some question's shortest path is in two hops
                        new_cids.append(torch.tensor([self.cid_pad], device=self._device))
                        new_cid_idxs.append(None)
                    else:
                        new_cids.append(_tail_cids[_new_cid_idxs])
                        new_cid_idxs.append(_new_cid_idxs)
                # logging.info(f"{head_cids.shape=} {head_concept_emb.shape=}")
            else:
                new_cids = curr_top_routing_cids
                new_cid_idxs = curr_top_routing_cid_idxs

            head_cids = pad_sequence(new_cids, batch_first=True, padding_value=self.cid_pad)
            for i, _cids in enumerate(head_cids):
                visited_cids[i].update(_cids.tolist())
            head_concept_emb = self.get_new_concept_emb(head_cids, new_cid_idxs, tail_concept_emb)

        # if self._vis_dict:
        #     self._vis_dict["all_top_routing_cids"] = all_top_routing_cids

        final_decoding_cids = [[] for _ in range(self._batch_size)]
        if self.filter_concepts_by_routing_score:
            for i in range(self._batch_size):
                _cids = torch.tensor(all_top_routing_cids[i])
                _concept_scores = torch.cat(all_top_routing_concept_scores[i])
                _routing_scores = torch.cat(all_top_routing_scores[i])

                assert _cids.shape == _concept_scores.shape
                assert _concept_scores.shape == _routing_scores.shape

                if compute_loss or compute_metrics:
                    _labels = torch.cat(all_top_routing_cid_labels[i])
                    assert _labels.shape == _cids.shape

                _, _routing_topk_idx = torch.topk(_routing_scores, min(top_n_routing_concepts, _cids.shape[0]))

                top_routing_concepts_mask = torch.full_like(_cids, False, device=self._device, dtype=torch.bool)
                top_routing_concepts_mask[_routing_topk_idx] = True

                if compute_loss or compute_metrics:
                    top_routing_concepts_mask[_labels] = True

                _concept_scores = _concept_scores[top_routing_concepts_mask]

                if compute_loss:
                    # cids_from_iteration = sorted(_cids[_labels].tolist())
                    # a_cids = a_concept_ids[i]
                    # _a_cids = a_cids[a_cids != self.cid_pad]
                    # cids_from_answer = sorted(_a_cids.tolist())
                    # print(f"{cids_from_iteration=}")
                    # print(f"{cids_from_answer=}")

                    _labels = _labels[top_routing_concepts_mask].to(torch.float)
                    positive_cids += _labels.sum().item()
                    total_cids += len(_labels)
                    concept_loss += self.concept_loss_fn(_concept_scores, _labels)

                _, _topk_idx = torch.topk(_concept_scores, min(top_n_decoding_concepts, len(_concept_scores)))
                _best_cids = _cids[_topk_idx].tolist()
                final_decoding_cids[i].extend(_best_cids)
        else:
            for i, (_cids, _concept_scores) in enumerate(zip(all_top_decoding_cids, all_top_decoding_cid_probs)):
                _, _topk_idx = torch.topk(torch.tensor(_concept_scores), min(top_n_decoding_concepts, len(_concept_scores)))
                _best_cids = torch.tensor(_cids)[_topk_idx].tolist()
                final_decoding_cids[i].extend(_best_cids)

        final_answers = []
        if do_decode:
            input_ids, position_ids, new_attention_mask = self.prepare_input_ids(q_input_ids, final_decoding_cids, a_input_ids)

            # inputs_embeds = self.transformer_embedding_layer(input_ids)
            if compute_loss:
                attention_mask = torch.cat((q_attention_mask, new_attention_mask), dim=1)
                output = self.transformer(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                max_answer_len = a_input_ids.shape[1]
                logits = output.logits[:,-max_answer_len-1:-1] # last token is <eos>
                labels = input_ids[:,-max_answer_len:]
                # logger.info(f"{logits.shape} {labels.shape}")
                decoding_loss = batched_decoding_loss(self.lm_loss_fn, logits, labels, pad_token_id=self.tok_pad)
            else:
                attention_mask = new_attention_mask
                generation_output = self.transformer.generate(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    max_length=512,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                input_max_len = input_ids.shape[1]
                outputs = generation_output.sequences[:,input_max_len:]
                scores = torch.stack(generation_output.scores)
                answer_strs = decode_answers(outputs, self.tokenizer)
                scores = score_by_generator(outputs, scores, self.tokenizer)
                for j in range(self._batch_size):
                    final_answers.append({
                        "decoding_cids": final_decoding_cids[j],
                        "answer": answer_strs[j],
                        "score": scores[j]
                    })


        # calculate recall of gold concepts and gold edges
        concept_precision = None
        concept_recall = None
        edge_precision = None
        edge_recall = None
        if compute_metrics:
            concept_precision = [0. for _ in range(self._batch_size)]
            concept_recall = [0. for _ in range(self._batch_size)]
            for i, (_pred_cids, _gold_set) in enumerate(zip(final_decoding_cids, a_cid_set)):
                # _a_cids = _a_cids[_a_cids != self.cid_pad]
                _pred_set = set(_pred_cids)
                _intersection = _pred_set.intersection(_gold_set)
                concept_recall[i] = len(_intersection) / len(_gold_set)
                concept_precision[i] = len(_intersection) / len(_pred_set)
                if verbose:
                    logger.info(f"{i} cids got {','.join(self.id2concept[c] for c in _pred_set)}")
                    logger.info(f"{i} expected {','.join(self.id2concept[c] for c in _gold_set)}")

            edge_precision = [0. for _ in range(self._batch_size)]
            edge_recall = [0. for _ in range(self._batch_size)]
            for i, (_bridge_cids, _gold_tails) in enumerate(zip(all_top_routing_cids, gold_tails)):
                _bridge_cid_set = set(_bridge_cids)
                _mask = torch.ne(_gold_tails, self.cid_pad)
                _gold_edge_set = set(_gold_tails[_mask].tolist())
                _intersection = _bridge_cid_set.intersection(_gold_edge_set)
                edge_recall[i] = len(_intersection) / max(len(_gold_edge_set), 1)
                edge_precision[i] = len(_intersection) / max(len(_bridge_cid_set), 1)

        return {
            "edge_loss": edge_loss,
            "concept_loss": concept_loss,
            "decoding_loss": decoding_loss,
            "concept_precision": concept_precision,
            "concept_recall": concept_recall,
            "edge_precision": edge_precision,
            "edge_recall": edge_recall,
            "positive_cids": positive_cids,
            "total_cids": total_cids,
            "positive_rate": positive_cids / total_cids if total_cids != 0 else 0.,
            "final_decoding_cids": final_decoding_cids,
            "final_answers": final_answers,
            "num_rough_recovered_edges": total_num_rough_recovered,
            "num_gold_truth_edges": total_num_gold_truth,
            "vis": self._vis_dict
        }

    def expand_graph(self, head_cids, visited_cids, gold_head_cids=None,
                     gold_tail_cids=None, teacher_forcing=True,
                     max_num_edges=1200):
        new_tail_cids = []
        new_heads = []
        new_tails = []
        new_edge_labels = []

        for i, _concepts in enumerate(head_cids):
            cid_mask = torch.ne(_concepts, self.cid_pad)

            _head_cids = _concepts[cid_mask].tolist()
            cid2headidx = {_cid: j for j, _cid in enumerate(_head_cids)}
            # prev_dists = _dists[cid_mask].tolist()

            head_cid_set = set(_head_cids)

            new_edges = set((head, tail)
                for head in head_cid_set for tail in self.kg_full.neighbors(head) \
                if tail not in head_cid_set and tail not in visited_cids[i]
            )

            if len(new_edges) == 0:
                new_tail_cids.append(torch.tensor([self.cid_pad]))
                new_heads.append(torch.tensor([self.cid_pad]))
                new_tails.append(torch.tensor([self.cid_pad]))
                new_edge_labels.append(torch.tensor([float(self.cid_pad)]))
                continue

            if self._vis_dict:
                raw_cand_tail_count = len(set(t for h, t in new_edges))
                self._vis_dict["raw_cand_tail_counts"][i].append(raw_cand_tail_count)

            if teacher_forcing:
                new_gold_edge_mask = \
                    torch.tensor([gold_head in head_cid_set and gold_head != self.cid_pad
                                  for gold_head in gold_head_cids[i].tolist()],
                                  device=self._device, dtype=torch.bool)
                new_gold_heads = gold_head_cids[i][new_gold_edge_mask].tolist()
                new_gold_tails = gold_tail_cids[i][new_gold_edge_mask].tolist()
                new_gold_edge_set = set(zip(new_gold_heads, new_gold_tails))
                max_num_new_edges = max_num_edges - len(new_gold_edge_set)

                # limit the total number of new edges
                if len(new_edges) > max_num_new_edges:
                    new_edges = list(new_edges)
                    idxs = np.random.default_rng().permutation(len(new_edges))
                    new_edges = set(new_edges[i] for i in idxs[:max_num_new_edges])

                # add in gold edges for training
                new_edges = new_edges | new_gold_edge_set
                new_edges = list(new_edges)
            else:
                new_edges = list(new_edges)
                if len(new_edges) > max_num_edges:
                    idxs = np.random.default_rng().permutation(len(new_edges))
                    new_edges = [new_edges[i] for i in idxs[:max_num_edges]]


            tail_cids = set(e[1] for e in new_edges)
            assert len(tail_cids.intersection(head_cid_set)) == 0

            tail_cids = list(tail_cids)
            cid2tailidx = {c: i for i, c in enumerate(tail_cids)}
            tail_cids = torch.tensor(tail_cids)
            new_tail_cids.append(tail_cids)
            new_heads.append(torch.tensor([cid2headidx[e[0]] for e in new_edges]))
            new_tails.append(torch.tensor([cid2tailidx[e[1]] for e in new_edges]))

            if teacher_forcing:
                new_edge_labels.append(torch.tensor([1. if e in new_gold_edge_set else 0. for e in new_edges]))

        new_tail_cids = pad_sequence(new_tail_cids, batch_first=True, padding_value=self.cid_pad).to(self._device)
        new_heads = pad_sequence(new_heads, batch_first=True, padding_value=self.cid_pad).to(self._device)
        new_tails = pad_sequence(new_tails, batch_first=True, padding_value=self.cid_pad).to(self._device)
        if teacher_forcing:
            new_edge_labels = pad_sequence(new_edge_labels, batch_first=True, padding_value=self.cid_pad).to(self._device)

        return new_tail_cids, new_heads, new_tails, new_edge_labels

    def expand_graph_with_rough_filtering(
            self, head_cids, visited_cids, q_emb, gold_head_cids=None,
            gold_tail_cids=None, teacher_forcing=True, max_num_edges=1200):

        all_new_edges = []
        all_cid2headidx = []
        all_head_cid_sets = []

        for i, _concepts in enumerate(head_cids):
            cid_mask = torch.ne(_concepts, self.cid_pad)

            _head_cids = _concepts[cid_mask].tolist()
            all_cid2headidx.append({_cid: j for j, _cid in enumerate(_head_cids)})
            # prev_dists = _dists[cid_mask].tolist()
            head_cid_set = set(_head_cids)
            all_head_cid_sets.append(head_cid_set)
            new_edges = set((head, tail)
                for head in head_cid_set for tail in self.kg_full.neighbors(head) \
                if tail not in head_cid_set and tail not in visited_cids[i]
            )

            all_new_edges.append(list(new_edges))

        # do rough filtering - find dot product between
        new_edge_lens = [len(_edges) for _edges in all_new_edges]
        print(f"Number of new edges: {new_edge_lens}")
        if sum(new_edge_lens) == 0:
            return None, None, None, None, []
        with torch.no_grad():
            batch_ids = torch.tensor([i for i, _edges in enumerate(all_new_edges) for _ in _edges], device=self._device)
            tail_ids = torch.tensor([t for _edges in all_new_edges for (h, t) in _edges], device=self._device).unsqueeze(0)
            tail_input_ids = self.tokenize_concept_ids(tail_ids) # [1, num_tails, max_concept_len]
            tail_tok_emb = self.transformer_embedding_layer(tail_input_ids) # (1, num_tails, max_concept_len, embed_dim)
            tail_emb = tail_tok_emb.sum(dim=2).squeeze(0) # (num_tails, embed_dim)
            _q_emb = q_emb[batch_ids] # (num_tails, embed_dim)
            rough_scores = torch.matmul(_q_emb.unsqueeze(1), tail_emb.unsqueeze(2)).view(tail_emb.shape[0])

        all_filtered_new_edges = []
        j = 0
        for i, l in enumerate(new_edge_lens):
            if l == 0:
                all_filtered_new_edges.append([])
                continue
            _scores = rough_scores[j:j+l]
            _, top_rough_score_idx = torch.topk(_scores, min(max_num_edges, l))
            all_filtered_new_edges.append([all_new_edges[i][k.item()] for k in top_rough_score_idx])
            j += l

        new_tail_cids = []
        new_heads = []
        new_tails = []
        new_edge_labels = []
        num_gold_truth_edges = 0
        num_rough_recovered_edges = 0

        for i, _concepts in enumerate(head_cids):
            new_edges = set(all_filtered_new_edges[i])
            head_cid_set = all_head_cid_sets[i]
            if teacher_forcing:
                new_gold_edge_mask = \
                    torch.tensor([gold_head in head_cid_set and gold_head != self.cid_pad
                                  for gold_head in gold_head_cids[i].tolist()],
                                  device=self._device, dtype=torch.bool)
                new_gold_heads = gold_head_cids[i][new_gold_edge_mask].tolist()
                new_gold_tails = gold_tail_cids[i][new_gold_edge_mask].tolist()
                new_gold_edge_set = set(zip(new_gold_heads, new_gold_tails))
                print(f"{i} number of gold edges {len(new_gold_edge_set)}")
                print(f"{i} number of unfiltered edges {len(all_new_edges[i])}")
                print(f"{i} number of filtered edges {len(new_edges)}")

                # count = 0
                # for h, t in new_gold_edge_set:
                #     if count >= 30: break
                #     print(f"{i} Gold edge ({self.id2concept[h]}, {self.id2concept[t]}) in new edges? {(h, t) in new_edges}")
                #     count += 1
                intersection = new_edges.intersection(new_gold_edge_set)
                num_gold_truth_edges += len(new_gold_edge_set)
                num_rough_recovered_edges += len(intersection)

                print(f"{i} number of gold edges in filtered set {len(intersection)}")

                new_edges = new_edges | new_gold_edge_set
                new_edges = list(new_edges)


            tail_cids = set(e[1] for e in new_edges)
            assert len(tail_cids.intersection(head_cid_set)) == 0

            tail_cids = list(tail_cids)
            cid2headidx = all_cid2headidx[i]
            cid2tailidx = {c: j for j, c in enumerate(tail_cids)}
            tail_cids = torch.tensor(tail_cids)
            new_tail_cids.append(tail_cids)
            new_heads.append(torch.tensor([cid2headidx[e[0]] for e in new_edges]))
            new_tails.append(torch.tensor([cid2tailidx[e[1]] for e in new_edges]))

            if teacher_forcing:
                new_edge_labels.append(torch.tensor([1. if e in new_gold_edge_set else 0. for e in new_edges]))

        new_tail_cids = pad_sequence(new_tail_cids, batch_first=True, padding_value=self.cid_pad).to(self._device)
        new_heads = pad_sequence(new_heads, batch_first=True, padding_value=self.cid_pad).to(self._device)
        new_tails = pad_sequence(new_tails, batch_first=True, padding_value=self.cid_pad).to(self._device)
        if teacher_forcing:
            new_edge_labels = pad_sequence(new_edge_labels, batch_first=True, padding_value=self.cid_pad).to(self._device)

        return new_tail_cids, new_heads, new_tails, new_edge_labels, num_rough_recovered_edges, num_gold_truth_edges

    def tokenize_concept_ids(self, concept_ids):
        batch_size, max_num_ids = concept_ids.shape

        concept_input_ids = [torch.tensor(self.id2concepttoks[c.item()]) if c != self.cid_pad else torch.tensor([self.tok_pad]) for cs in concept_ids for c in cs]
        concept_input_ids = pad_sequence(concept_input_ids, batch_first=True, padding_value=self.tok_pad).to(self._device)
        # (batch_size * max_num_ids, max_concept_len)
        concept_input_ids = concept_input_ids.view(batch_size, max_num_ids, self.cid_pad)
        return concept_input_ids # (batch_size, max_num_concepts, max_concept_len)

    def encode_concepts(self, concept_ids, distances, q_hidden_states, q_attention_mask):
        # new_concept_id_masks = [torch.tensor([c.item() not in self.visited for c in cs], device=self._device) for cs in concept_ids]
        # new_concept_ids = [cs[mask] for cs, mask in zip(concept_ids, new_concept_id_masks)]
        # new_concept_distances = [ds[mask] for ds, mask in zip(distances, new_concept_id_masks)]
        # new_concept_ids = pad_sequence(new_concept_ids, batch_first=True, padding_value=-1)
        # new_concept_distances = pad_sequence(new_concept_distances, batch_first=True, padding_value=0)

        concept_input_ids = self.tokenize_concept_ids(concept_ids)

        concept_tok_emb = self.transformer_embedding_layer(concept_input_ids)
        # (batch_size, max_num_concepts, max_concept_len, embed_dim)
        batch_size, max_num_concepts, max_concept_len, embed_dim = concept_tok_emb.shape

        biattn_embs, _ = self.biatt(
            concept_tok_emb.view(-1, max_concept_len, embed_dim),
            q_hidden_states.repeat_interleave(max_num_concepts, 0),
            q_attention_mask.repeat_interleave(max_num_concepts, 0))

        biattn_embs = biattn_embs.view(batch_size, -1, max_concept_len, embed_dim)
        # H_c^con: (batch_size, max_num_concepts, max_concept_len, embed_dim)

        concept_emb = torch.cat((concept_tok_emb, biattn_embs), dim=-1)
        # [H_c^tok; H_c^con] (batch_size, max_num_concepts, max_concept_len, embed_dim * 2)

        mask = torch.ne(concept_input_ids, self.tok_pad).unsqueeze(3).expand(concept_emb.shape).float()
        concept_emb = torch.max(concept_emb * mask, 2)[0]
        # max[H_c^tok; H_c^con] (batch_size, max_num_concepts, embed_dim * 2)

        # node_repr = relu(self.attn_2(enc_memory))
        dist_emb = self.dist_embd(distances)
        relu = nn.ReLU()
        concept_emb = torch.cat((relu(self.concept_emb_lin(concept_emb)), dist_emb), dim=-1)
        # h_c (batch_size, max_num_concepts, embed_dim * 2)

        # # save new concept embeddings to memory
        # assert concept_emb.shape[0] * concept_emb.shape[1] == concept_ids.shape[0] * concept_ids.shape[1]
        # flat_new_concept_emb = concept_emb.view(concept_emb.shape[0] * concept_emb.shape[1], -1)
        #
        # # retrieve concept embeddings from memory
        # self.visited.update({cid: flat_new_concept_emb[i] for i, cid in enumerate(new_concept_ids.view(-1)) if cid != self.cid_pad})
        # flat_concept_ids = concept_ids.view(-1) #(batch_size * max_num_concepts)
        # concept_emb = torch.stack([self.visited[cid] for cid in flat_concept_ids], dim=0)
        # concept_emb = concept_emb.view(batch_size, concept_ids.shape[1], -1

        return concept_emb #(batch_size, max_num_concepts, embed_dim * 2)

    def encode_relations(self, head_cids, tail_cids, heads, tails, gold_heads, gold_tails):
        # retrieve relations from conceptnet
        rlns = torch.full((self._batch_size, heads.shape[1], self.max_num_rlns_per_edge), self.rln_pad)
        for i, (_h_cids, _t_cids, _h_idxs, _t_idxs) in enumerate(zip(head_cids, tail_cids, heads, tails)):
            h_idxs = _h_idxs[_h_idxs != self.cid_pad]
            t_idxs = _t_idxs[_t_idxs != self.cid_pad]
            _heads = _h_cids[h_idxs].tolist()
            _tails = _t_cids[t_idxs].tolist()
            for j, (h, t) in enumerate(zip(_heads, _tails)):
                if h != self.cid_pad:
                    _rlns = [rel_dict['rel'] for rel_dict in self.kg_full[h][t].values()][:self.max_num_rlns_per_edge]
                    rlns[i,j,:len(_rlns)] = torch.tensor(_rlns)

        # logger.info(f"Unfound edges: {unfound_edges}, total_edges {total_edges}, unfound edge is gold: {unfound_edge_is_gold}")
        rlns = rlns.to(self._device) # (batch_size, max_num_edges, max_num_rlns)
        rln_embs = self.relation_embd(rlns)
        mask = torch.ne(rlns, self.rln_pad)
        rln_embs = self.relation_embd(rlns).masked_fill(mask.unsqueeze(-1).expand_as(rln_embs), 0.)

        # average rln embeddings if an edge has multiple rlns
        rln_embs = rln_embs.sum(dim=2) # (batch_size, max_num_edges, embed_dim)
        num_rels = mask.sum(dim=2).clamp(min=1.).unsqueeze(-1) # (batch_size, max_num_edges, 1)
        # logger.info(f"num_rels.shape {num_rels.shape}, rln_embs.shape {rln_embs.shape}")
        rln_embs = rln_embs / num_rels # (batch_size, max_num_edges, embed_dim)

        return rln_embs

    def encode_triples(self, head_concept_emb, relation_emb, tail_concept_emb, heads, tails):
        _heads, _tails = heads.clamp(min=0), tails.clamp(min=0) # deal with -1 idxs
        # logger.info(f"head_concept_emb.shape {head_concept_emb.shape}, rln_emb.shape {relation_emb.shape}")
        # logger.info(f"tail_concept_emb.shape {tail_concept_emb.shape} heads.shape {heads.shape}")
        head_repr = torch.gather(head_concept_emb, 1, _heads.unsqueeze(-1).expand(self._batch_size, heads.shape[1], head_concept_emb.shape[-1]))
        tail_repr = torch.gather(tail_concept_emb, 1, _tails.unsqueeze(-1).expand(self._batch_size, tails.shape[1], tail_concept_emb.shape[-1]))
        return torch.cat((head_repr, relation_emb, tail_repr), dim=-1)

    def prepare_input_ids(self, q_input_ids, final_decoding_cids, a_input_ids):
        # logging.info(f"{q_input_ids.shape=}, {a_input_ids.shape=}, {batch_size=}")
        assert q_input_ids.shape[0] == self._batch_size
        if a_input_ids is not None: assert a_input_ids.shape[0] == self._batch_size
        concept_input_ids = [[] for _ in range(self._batch_size)]
        position_id_offsets = [[] for _ in range(self._batch_size)]

        for i in range(len(final_decoding_cids)):
            for _cid in final_decoding_cids[i]:
                ids = self.tokenizer.encode(self.id2concept[_cid].replace("_", " "), add_special_tokens=False)
                concept_input_ids[i].extend(ids)
                position_id_offsets[i].extend(range(len(ids)))

        concept_input_ids = [torch.tensor(_cids) for _cids in concept_input_ids]
        position_id_offsets = [torch.tensor(_pids) for _pids in position_id_offsets]

        # assume q_input_ids and a_input_ids are already padded
        sep_tokens = torch.full((self._batch_size, 1), self.tokenizer.sep_token_id, device=self._device)
        concept_input_ids = pad_sequence(concept_input_ids, batch_first=True, padding_value=self.tok_pad).to(self._device)
        position_id_offsets = pad_sequence(position_id_offsets, batch_first=True, padding_value=0)
        # eos_tokens = torch.full((self._batch_size, 1), self.tokenizer.eos_token_id, device=self._device)

        position_ids = get_gpt2_position_ids(q_input_ids, position_id_offsets, a_input_ids, include_q=False).to(self._device)

        if a_input_ids is None:
            input_ids = torch.cat((sep_tokens,
                                   concept_input_ids,
                                   sep_tokens), dim=1)
        else:
            input_ids = torch.cat((sep_tokens,
                                   concept_input_ids,
                                   sep_tokens,
                                   a_input_ids), dim=1)
        attention_mask = torch.where(input_ids != self.tokenizer.pad_token_id, 1., 0.).to(self._device)

        return input_ids, position_ids, attention_mask

    def get_new_concept_emb(self, new_head_cids, new_head_cid_idxs, prev_tail_concept_emb):
        head_concept_emb = []
        max_num_cids = new_head_cids.shape[1]
        concept_emb_dim = prev_tail_concept_emb.shape[-1]
        for _tail_concept_emb, _idxs in zip(prev_tail_concept_emb, new_head_cid_idxs):
            if _idxs is None:
                head_concept_emb.append(torch.zeros(max_num_cids, concept_emb_dim, device=self._device))
            else:
                _head_concept_emb = _tail_concept_emb[_idxs]
                _pad = torch.zeros(max_num_cids - _head_concept_emb.shape[0], concept_emb_dim, device=self._device)
                # logging.info(f"{_tail_concept_emb.shape=}, {_idxs.shape=}, {_head_concept_emb.shape=}, {_pad.shape=}")
                head_concept_emb.append(torch.cat((_head_concept_emb, _pad), dim=0))
        return torch.stack(head_concept_emb, dim=0)

    @torch.no_grad()
    def _visualize_graph_expansion(self, hop, head_cids, tail_cids, heads, tails, gold_heads, gold_tails):
        max_tuples_to_viz = 18
        print("head_cids", head_cids)
        print("tail_cids", tail_cids)
        print("heads", heads)
        print("tails", tails)
        logger.info(f"------- Visualizing graph expansion at hop {hop} -------")
        for i, (_hcids, _tcids, _hs, _ts, _ghs, _gts) in enumerate(zip(head_cids, tail_cids, heads, tails, gold_heads, gold_tails)):
            hcids = _hcids[_hcids != self.cid_pad]
            tcids = _tcids[_tcids != self.cid_pad]
            hs = _hs[_hs != self.cid_pad]
            all_hcids = _hcids[hs].tolist()
            ts = _ts[_ts != self.cid_pad]
            all_tcids = _tcids[ts].tolist()
            ghs = _ghs[_ghs != self.cid_pad].tolist()
            all_h_strs = [self.id2concept[hcid] for hcid in all_hcids]
            all_t_strs = [self.id2concept[tcid] for tcid in all_tcids]
            logger.info(f"--- ex[{i}]: {len(all_h_strs)} triples, {len(ghs)} gold triples, {len(hcids)} heads, {len(tcids)} tails")

            if max_tuples_to_viz:
                all_h_strs = all_h_strs[:max_tuples_to_viz]
                all_t_strs = all_t_strs[:max_tuples_to_viz]
            log_str = ""
            gold_hts = set(zip(_ghs, _gts))
            for j, (_hstr, _tstr) in enumerate(zip(all_h_strs, all_t_strs)):
                if (_hstr, _tstr) in gold_hts:
                    log_str += f"<{_hstr:10.10}=={_tstr:10.10}> "
                else:
                    log_str += f"({_hstr:10.10}, {_tstr:10.10}) "
                if (j+1) % 3 == 0:
                    print(log_str)
                    log_str = ""




class BiAttention(nn.Module):
    def __init__(self, input_dim, memory_dim, hid_dim, dropout):
        super(BiAttention, self).__init__()
        self.dropout = dropout
        self.input_linear_1 = nn.Linear(input_dim, 1, bias=False)
        self.memory_linear_1 = nn.Linear(memory_dim, 1, bias=False)

        self.input_linear_2 = nn.Linear(input_dim, hid_dim, bias=True)
        self.memory_linear_2 = nn.Linear(memory_dim, hid_dim, bias=True)

        self.dot_scale = np.sqrt(input_dim)

    def forward(self, input, memory, mask):
        """
        :param input: context_encoding N * Ld * d
        :param memory: query_encoding N * Lm * d
        :param mask: query_mask N * Lm
        :return:
        """
        # print(input.size())
        # print(memory.size())
        # print(mask.size())
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = F.dropout(input, self.dropout,training=self.training)  # N x Ld x d
        memory = F.dropout(memory, self.dropout,training=self.training)  # N x Lm x d

        input_dot = self.input_linear_1(input)  # N x Ld x 1
        memory_dot = self.memory_linear_1(memory).view(bsz, 1, memory_len)  # N x 1 x Lm
        # N * Ld * Lm
        cross_dot = torch.bmm(input, memory.permute(0, 2, 1).contiguous()) / self.dot_scale
        # [f1, f2]^T [w1, w2] + <f1 * w3, f2>
        # (N * Ld * 1) + (N * 1 * Lm) + (N * Ld * Lm)
        att = input_dot + memory_dot + cross_dot  # N x Ld x Lm
        # N * Ld * Lm
        att = att - 1e30 * (1 - mask[:, None].float())

        input = self.input_linear_2(input)
        memory = self.memory_linear_2(memory)

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1,
                                                                input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input * output_one, output_two * output_one],dim=-1), memory
