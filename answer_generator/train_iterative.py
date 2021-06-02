from __future__ import absolute_import, division, print_function
import json
import configargparse
import os
import shutil
import random
import math
import csv
from pprint import pprint

import logging


import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from model import IterativePathFinder
from dataset import IterativePathFinderDatahelper
from argparser import get_iterative_argparser
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logger = logging.getLogger()
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

def list2str(list):
    return " ".join([str(x) for x in list])

def str2list(str):
    return [int(x) for x in str.split(" ")]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def set_log(debug=False, log_file=None):
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter('[%(asctime)s] %(message)s', '%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file != None:
        logfile = logging.FileHandler(log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)



sign_list = {
    "edge_loss" : 1.,
    "concept_loss" : 1.,
    "decoding_loss" : 1.,
    "concept_recall" : -1.,
    "concept_precision" : -1.,
    "concept_f1" : -1.
} # 1: minimize, -1: maximize

def train(args, datahelper, model):
    """ Train the model """
    tb_log_dir = os.path.join(args.output_dir, "tensorboard/")
    writer = SummaryWriter(log_dir=tb_log_dir)

    train_dataset = datahelper.trainset
    args.train_batch_size = args.batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
        drop_last=False, collate_fn=train_dataset.collate_fn
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // len(train_dataloader) + 1
    else:
        t_total = len(train_dataloader) * args.num_train_epochs

    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    if args.validate_per_epoch != 0:
        # overwrite validate steps
        args.validate_steps = int(steps_per_epoch / args.validate_per_epoch)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_ratio * t_total), num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info(f"  Evaluation steps = {args.validate_steps} / {steps_per_epoch}")

    # best_valid = {'bleu':0.0, 'ppl':1e6, 'acc':0.0}
    
    global_step = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    best_valid = {k: v * math.inf for k, v in sign_list.items() }

    for epoch in train_iterator:
        local_step = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            if not 390 <= step < 392 : continue
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            model.train()
            output_dict = model(
                **batch,
                teacher_forcing=True, do_decode=args.do_decode, compute_loss=True,
                top_n_decoding_concepts=args.train_top_n_decoding_concepts,
                top_n_routing_concepts=args.train_top_n_routing_concepts,
                max_num_edges=args.train_max_num_edges,
                verbose=False
            )

            loss = output_dict["edge_loss"] * args.edge_loss_weight
            if epoch >= args.concept_cls_delay_epochs:
                loss += output_dict["concept_loss"] * args.concept_loss_weight
            if args.do_decode:
                loss += output_dict["decoding_loss"] * args.decoding_loss_weight

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            local_step += 1

            # log statistics
            writer.add_scalar('train/total_loss', loss.item(), global_step)
            writer.add_scalar('train/edge_loss', output_dict["edge_loss"].item(), global_step)
            writer.add_scalar('train/concept_loss', output_dict["concept_loss"].item(), global_step)
            if args.do_decode:
                writer.add_scalar('train/decoding_loss', output_dict["decoding_loss"].item(), global_step)

            writer.add_scalar('train/positive_rate', output_dict["positive_rate"], global_step)

            avg_concept_f1 = -1.
            if len(output_dict["concept_precision"]) == 0 or len(output_dict["concept_recall"]) == 0:
                logger.warning(f"Got empty array for concept precision in batch {step}")
            else:
                concept_precision, concept_recall = output_dict["concept_precision"], output_dict["concept_recall"]
                edge_precision, edge_recall = output_dict["edge_precision"], output_dict["edge_recall"]
                avg_concept_precision = sum(concept_precision) / len(concept_precision)
                avg_concept_recall = sum(concept_recall) / len(concept_recall)
                avg_edge_precision = sum(edge_precision) / len(edge_precision)
                avg_edge_recall = sum(edge_recall) / len(edge_recall)

                avg_concept_f1 = sum(p*r / (p+r) if p+r != 0 else 0. for p, r in zip(concept_precision, concept_recall)) / len(concept_precision)
                writer.add_scalar('train/concept_precision', avg_concept_precision, global_step)
                writer.add_scalar('train/concept_recall', avg_concept_recall, global_step)
                writer.add_scalar('train/concept_f1', avg_concept_f1, global_step)

                writer.add_scalar('train/edge_precision', avg_edge_precision, global_step)
                writer.add_scalar('train/edge_recall', avg_edge_recall, global_step)

                if global_step % args.logging_steps == 0:
                    logger.info(f"St {global_step} | L={loss.item():.2f} | lm_L={output_dict['decoding_loss'].item():.2f} | cpt_R={avg_concept_recall:.2%} | edge_R={avg_edge_recall:.2%}")

            # evaluate and save checkpoint
            if (step +1) % args.validate_steps == 0:
                result = evaluate(args, datahelper, model, args.do_decode)
                logger.info(f"Epoch: {epoch} | dev {args.evaluate_metrics}: {result[args.evaluate_metrics]:.4f}")
                # writer.add_scalar(f"dev/{args.evaluate_metrics}", result[args.evaluate_metrics], global_step)

                for k, v in result.items():
                    writer.add_scalar(f"dev/{k}", v, global_step)

                if args.evaluate_metrics != "concept_recall" or (args.evaluate_metrics == "concept_recall" and epoch >= args.concept_cls_delay_epochs):
                    if (result[args.evaluate_metrics] - best_valid[args.evaluate_metrics]) * sign_list[args.evaluate_metrics] < 0:
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

                        model_save_path = os.path.join(args.output_dir, args.model_save_name)
                        torch.save(model_to_save.state_dict(), model_save_path)
                        logger.info("Saving model checkpoint to %s", model_save_path)
                        best_valid[args.evaluate_metrics] = result[args.evaluate_metrics]

@torch.no_grad()
def evaluate(args, datahelper, model, verbose=False, pred_out_file=None):
    if args.eval_crowdsourced:
        eval_dataset = datahelper.crowdsourced_devset
    else:
        eval_dataset = datahelper.devset

    tokenizer = eval_dataset.tokenizer
    generate_only = args.generate_only or args.generate_with_teacher_forcing
    if args.generate_only and args.generate_with_teacher_forcing:
        raise ValueError("Cannot set both `generate_only` and `generate_with_teacher_forcing`")

    do_decode = args.do_decode

    if generate_only: assert do_decode and pred_out_file is not None
    compute_loss = not generate_only
    compute_metrics = not generate_only
    teacher_forcing = (not generate_only) or args.generate_with_teacher_forcing

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
        drop_last=False, collate_fn=eval_dataset.collate_fn)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    pred_out_f = None
    pred_writer = None
    qid2qstr = {}
    if pred_out_file:
        pred_out_f = open(pred_out_file, "w")
        fieldnames = ["qid", "question", "gold answer", "pred answer", "score", "gold acs", "pred acs"]
        pred_writer = csv.DictWriter(pred_out_f, fieldnames=fieldnames, delimiter="\t")
        pred_writer.writeheader()


    res_dict = {
        metric: 0. for metric in ["edge_loss", "concept_loss",
                                  "concept_recall", "concept_precision",
                                  "edge_recall", "edge_precision"]
    }

    if do_decode: res_dict["decoding_loss"] = 0.

    for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        batch = {
            k: v.to(args.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if generate_only:
            input_args = {"qids", "q_input_ids", "q_concept_ids"}
            if teacher_forcing:
                input_args |= {"gold_heads", "gold_tails"}
            batch_for_model = {
                k: v for k, v in batch.items() if k in input_args
            } # do not give model gold answers and edges
        else:
            batch_for_model = batch

        #torch.autograd.set_detect_anomaly(True)
        output_dict = model(
            **batch_for_model,
            teacher_forcing=teacher_forcing,
            compute_loss=compute_loss,
            do_decode=do_decode,
            top_n_decoding_concepts=args.eval_top_n_decoding_concepts,
            top_n_routing_concepts=args.eval_top_n_routing_concepts,
            top_n_routing_tails=args.eval_top_n_routing_tails,
            max_num_edges=args.eval_max_num_edges,
            compute_metrics=compute_metrics,
            verbose=verbose
        )

        if pred_writer:
            for j in range(len(batch["qids"])):
                qid = batch["qids"][j]
                row_dict = {}
                if qid not in qid2qstr:
                    q_input_ids = eval_dataset.qid2q[qid]
                    qid2qstr[qid] = tokenizer.decode(q_input_ids)
                q_string = qid2qstr[qid]
                row_dict["qid"] = qid
                row_dict["question"] = q_string

                a_input_ids = batch["a_input_ids"][j]
                a_input_ids = a_input_ids[a_input_ids != tokenizer.pad_token_id]
                a_string = tokenizer.decode(a_input_ids)
                row_dict["gold answer"] = a_string

                row_dict["pred answer"] = output_dict["final_answers"][j]["answer"]
                row_dict["score"] = output_dict["final_answers"][j]["score"].item()

                gold_ac_strs = [eval_dataset.graph_info["i2e"][c] for c in batch["a_concept_ids"][j] if c != model.cid_pad]
                row_dict["gold acs"] = ",".join(gold_ac_strs)
                pred_ac_strs = [eval_dataset.graph_info["i2e"][c] for c in output_dict["final_decoding_cids"][j] if c != model.cid_pad]
                row_dict["pred acs"] = ",".join(pred_ac_strs)
                if i % 10 == 0 and j == 0:
                    pprint(row_dict)
                pred_writer.writerow(row_dict)

        if not generate_only:
            res_dict["edge_loss"] += output_dict["edge_loss"].item() * args.edge_loss_weight
            res_dict["concept_loss"] += output_dict["concept_loss"].item() * args.concept_loss_weight
            if do_decode:
                res_dict["decoding_loss"] += output_dict["decoding_loss"].item() * args.decoding_loss_weight
            for metric in ["concept_recall", "concept_precision", "edge_recall", "edge_precision"]:
                res_dict[metric] += sum(output_dict[metric])
    if not generate_only:
        for metric in res_dict:
            res_dict[metric] /= len(eval_dataset)

        # total_concept_f1 += sum(p*r / (p+r) if p+r != 0 else 0. for p, r in zip(output_dict["concept_precision"], output_dict["concept_recall"])) / len(output_dict["concept_recall"])
    if pred_out_f: pred_out_f.close()

    return res_dict

def generate(args, generator, model, dataset):
    model.eval()
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)
    total_hypos = []
    
    for i,batch in enumerate(tqdm(loader)):
        batch = tuple(t.to(args.device) for t in batch)
        sample = {
            "input_ids":batch[0],
            "attention_mask":batch[2],
            "position_ids":batch[3]
        }

        with torch.no_grad():
            hypos = generator.generate(model, sample, dataset)

        total_hypos.extend(hypos)
    return total_hypos

class JsonDumpHelper(json.JSONEncoder):
    def default(self, obj):
        if type(obj) != str:
            return str(obj)
        return json.JSONEncoder.default(self, obj)

def main():
    parser = get_iterative_argparser()
    args = parser.parse_args()

    if args.overwrite_output_dir:
       if os.path.exists(args.output_dir):
           shutil.rmtree(args.output_dir)

    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
        args.n_gpu = 1
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    args.device = device

    # Setup logging
    set_log(args.debug, os.path.join(args.output_dir, "log.txt"))

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer

    # config_class = transformers.GPT2Config.f
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    logger.info("Preparing dataset...")
    datahelper = IterativePathFinderDatahelper(
        tokenizer, args.preprocessed_dir, args.graph_info_dir, max_num_edges=args.train_max_num_edges // 2)
    gpt2_model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.resize_token_embeddings(len(tokenizer))
    logger.info(f"New tokenizer has {len(tokenizer)} tokens")

    if args.lm_load_path:
        gpt2_model.load_state_dict(torch.load(args.lm_load_path))

    model = IterativePathFinder(
        transformer_model=gpt2_model,
        tokenizer=tokenizer,
        graph_info=datahelper.graph_info,
        args=args
    )

    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' % json.dumps(vars(args), cls=JsonDumpHelper, indent=4, sort_keys=True))
    logger.info('-' * 100)

    if not args.eval_only:
        model.to(args.device)
        train(args, datahelper, model)
    else:
        pred_out_file = os.path.join(args.output_dir, "final_answers.tsv")
        if args.model_load_path is None:
            raise ValueError("Must provide model_load_path to evaluate model")
        model.load_state_dict(torch.load(args.model_load_path))
        model.to(args.device)
        result = evaluate(
            args, datahelper, model, pred_out_file=pred_out_file)
        print("----- Evaluation results -----")
        pprint(result)

    #     logger.info("Test evaluate {}: {:.4f}".format(args.evaluate_metrics, result[args.evaluate_metrics]))

if __name__ == '__main__':
    main()