import os
import math

import dataset
import numpy as np
import logging
import shutil
import pickle
import random

import configargparse
from configargparse import Namespace # to make copy of namespace

from tqdm import tqdm, trange

import json
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import transformers

from argparser import get_ansgen_args
from dataset import AnsgenTrainingDatahelper, AnswerGeneratorDataset
from utils import batched_decoding_loss
from multitask import MultiLinearTrainingSchedule, MultiTaskDataLoader

from evaluator import generate_answers, evaluate_answers

# torch.set_num_threads(4)

logger = logging.getLogger()

cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def setup_logger(args):
    # log file
    if args.num_epoch == 0:
        log_path = os.path.join(args.save_dir, 'test.log')
    else:
        log_path = os.path.join(args.save_dir, 'train.log')

    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path, 'w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info('args: {}'.format(args))


def copy_args(args):
    return Namespace(**vars(args))

def setup_multitask_dataloaders(args, tokenizer):
    """ Initial experiments --

    n5-w20-s0.05: Warmup for 20 subepochs (w20), training only on base,
        then increase linearly the ratio of examples with 5 noisy concepts by
        step size of 0.05 per subepoch (s0.05)

    n1,2,5-w10-s0.1-d10: Warmup for 10 subepochs (w10), training only on base,
        then introduce 1 noisy concept. After delay of 10 (d10),
        introduce 2 noisy concepts, then 5 after same delay.
    """
    mtype = args.multitask_type.split("-")

    base_args = copy_args(args)
    base_args.max_additional_concepts = 0
    base_datahelper = AnsgenTrainingDatahelper(tokenizer, base_args,  teacher_forcing=True)
    datahelpers = [base_datahelper]
    task_names = ["base"]

    mtype_dict = {
        "delay": 10,
        "final_ratios": [],
        "ratio_step_size": 0.05,
    }
    for item in mtype:
        if item.startswith("n"):
            mtype_dict["num_noisy_list"] = [int(x) for x in item[1:].split(",")]
        elif item.startswith("w"):
            mtype_dict["warmup_epochs"] = int(item[1:])
        elif item.startswith("s"):
            mtype_dict["ratio_step_size"] = float(item[1:])
        elif item.startswith("d"):
            mtype_dict["delay"] = [int(x) for x in item[1:].split(",")]
        elif item.startswith("r"):
            mtype_dict["final_ratios"] = [float(x) for x in item[1:].split(",")]

    if len(mtype_dict["final_ratios"]) == 0:
        mtype_dict["final_ratios"] = [1.0 for _ in mtype_dict["num_noisy_list"]]

    num_noisy_list = mtype_dict["num_noisy_list"]
    for num_noisy in num_noisy_list:
        logger.info(f"Constructing datahelper for {num_noisy} noisy concepts per example")
        _args = copy_args(args)
        _args.max_additional_concepts = num_noisy
        dh = AnsgenTrainingDatahelper(tokenizer, _args, teacher_forcing=True)
        datahelpers.append(dh)
        task_names.append(f"noisy_{num_noisy}")
        logger.info(f'    num train samples: {len(dh.trainset)}, steps: {math.ceil(len(dh.trainset)//args.batch_size)}')

    train_schedule = MultiLinearTrainingSchedule(
        base_dataset=base_datahelper.trainset,
        batch_size=args.batch_size,
        start_subepoch_list=[mtype_dict["warmup_epochs"] + i * mtype_dict["delay"] for i in range(len(num_noisy_list))],
        ratio_step_size_list=[mtype_dict["ratio_step_size"] for _ in range(len(num_noisy_list))],
        final_ratio_list=mtype_dict["final_ratios"],
        num_subepochs=args.multitask_subepochs
    )

    dataloaders = []
    for dh in datahelpers:
        dl = DataLoader(dh.trainset, shuffle=True, batch_size=args.batch_size,
                        collate_fn=dh.trainset.collate_fn)
        dataloaders.append(dl)

    # currently just assume last task is the most difficult task, evaluate on it
    eval_dh = datahelpers[-1]
    eval_dataloader = DataLoader(eval_dh.devset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=eval_dh.devset.collate_fn)
    final_dataloader = DataLoader(eval_dh.crowd_devset, shuffle=False, batch_size=args.batch_size,
                                  collate_fn=eval_dh.crowd_devset.collate_fn)

    multitask_dataloader = MultiTaskDataLoader(
        tasks=dataloaders,
        schedule_fn=train_schedule,
        return_task_name=False,
        task_names=task_names,
    )

    return multitask_dataloader, eval_dataloader, final_dataloader


def setup_data_and_model(args):
    logger.info('Loading data & model')

    if args.model_type == "bart":
        tokenizer = transformers.BartTokenizer.from_pretrained("facebook/bart-base")
        model = transformers.BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    elif args.model_type == "gpt2":
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
    else:
        raise ValueError(f"Invalid model type {args.model_type}")

    if args.multitask_type:
        train_dataloader, eval_dataloader, final_dataloader = setup_multitask_dataloaders(args, tokenizer)
    else:
        datahelper = AnsgenTrainingDatahelper(tokenizer, args, teacher_forcing=True)
        train_dataloader = DataLoader(datahelper.trainset, shuffle=True,
                                      batch_size=args.batch_size,
                                      collate_fn=datahelper.trainset.collate_fn)
        eval_dataloader = DataLoader(datahelper.devset, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=datahelper.devset.collate_fn)
        final_dataloader = DataLoader(datahelper.crowd_devset, shuffle=False,
                                      batch_size=args.generation_batch_size,
                                      collate_fn=datahelper.crowd_devset.collate_fn)

    model.resize_token_embeddings(len(tokenizer))

    # configs = transformers.GPT2Config.from_pretrained(args.model, cache_dir='../cache/')
    # tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.model, cache_dir='../cache/')
    # gpt = transformers.GPT2Model.from_pretrained(args.model, cache_dir='../cache/')
    # logger.info('Old vocab size: {}'.format(configs.vocab_size))
    # logger.info('Processing data')
    # datahelper = DataHelper(os.path.join('./data', args.data_dir), tokenizer=tokenizer)
    # configs.vocab_size = len(tokenizer)
    # logger.info('New vocab size: {}'.format(configs.vocab_size))
    # gpt.resize_token_embeddings(len(tokenizer))
    # model = GPT2LM(gpt, configs)
    model.to(args.device)
    return model, train_dataloader, eval_dataloader, final_dataloader, tokenizer


def run_training(model, train_dataloader, eval_dataloader, tokenizer, args):
    # ----------------------------------------------------- #
    # checkpoint directory
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    setup_logger(args)

    model_ckpt = os.path.join(args.save_dir, 'model.ckpt')

    writer = SummaryWriter(log_dir=args.save_dir)
    # ----------------------------------------------------- #
    # load data & init model and optimizer
    is_multitask = isinstance(train_dataloader, MultiTaskDataLoader)
    if not is_multitask:
        t_total = len(train_dataloader) * args.num_epoch
    else:
        t_total = sum(sum(train_dataloader.schedule_fn(e)) for e in range(args.num_epoch))
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #     ]
    optimizer = transformers.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # ----------------------------------------------------- #

    # training
    best_dev_loss = 1e19
    # train_iterator = trange(int(args.num_epoch), desc="Epoch")
    step_nogress = 0
    global_step = 0
    save_id = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_model_epoch = 0
    for epoch in range(args.num_epoch):
        train_loss = 0.0
        num_steps = 0
        model.train()
        if is_multitask:
            curr_schedule = train_dataloader.schedule_fn(epoch)
            epoch_total = sum(curr_schedule)
            schedule_str = ",".join(f"{task}={n}" for task, n in zip(train_dataloader.task_names, curr_schedule))
            print(f"Epoch {epoch} schedule: {schedule_str}")
        else:
            epoch_total = len(train_dataloader)
        epoch_iterator = tqdm(train_dataloader, desc=f"Train epoch {epoch}", total=epoch_total)
        for step, batch in enumerate(epoch_iterator):
            optimizer.zero_grad()
            task_name = None
            if isinstance(train_dataloader, MultiTaskDataLoader) and train_dataloader.return_task_name:
                batch, task_name = batch
            input_ids = batch["input_ids"].to(args.device)
            weights = batch["weights"].to(args.device)
            if args.model_type == "bart":
                decoder_input_ids = batch["decoder_input_ids"].to(args.device)

                output = model(input_ids=input_ids,
                               decoder_input_ids=decoder_input_ids,
                               return_dict=True)

                logits = output.logits[:, :-1]
                labels = decoder_input_ids[:, 1:]

            elif args.model_type == "gpt2":
                attention_mask = batch["attention_mask"].to(args.device)
                position_ids = batch.get("position_ids", None)
                if position_ids is not None:
                    position_ids = position_ids.to(args.device)

                output = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               return_dict=True)

                logits = output.logits[:,-batch["max_answer_len"]:-1]
                labels = input_ids[:,-batch["max_answer_len"]+1:]

            loss = batched_decoding_loss(cross_entropy_loss, logits, labels, weights,
                                         pad_token_id=tokenizer.pad_token_id)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule

            train_loss += loss.item()
            tr_loss += loss.item()
            num_steps += 1 # len(batch)
            log = 'Epoch: {:03d}, Iter: {:03d}, step loss: {:.4f}'
            if step % 100 == 0:
                logger.info(log.format(epoch, step, loss.item()))
            writer.add_scalar('Train/nll', loss.item(), global_step)

            if task_name:
                writer.add_scalar(f'Train/{task_name}', loss.item(), global_step)
            # writer.add_scalar('Train/nll_no_pad', loss_no_pad.item(), global_step)

            global_step += 1

        train_loss /= num_steps
        log = 'Epoch: {:03d} Train loss: {:.4f}'
        logger.info(log.format(epoch, train_loss))

        if (epoch+1) % args.eval_epochs == 0:
            result_dev = evaluation(model, eval_dataloader, tokenizer, args)
            log = 'Epoch: {:03d}, Dev ppl: {:.4f} loss: {:.4f}'
            if result_dev['loss'] <= best_dev_loss:
                logger.info("***************************************")
                logger.info(f"* Saved better model at epoch {epoch} *")
                logger.info("***************************************")
                best_dev_loss = result_dev['loss']
                torch.save(model.state_dict(), '{}'.format(model_ckpt))
                step_nogress = 0
                best_model_epoch = epoch

            logger.info(log.format(epoch, result_dev['ppl'], result_dev['loss']))
            writer.add_scalar('Dev/nll', result_dev['loss'], epoch)
            # writer.add_scalar('Dev/nll_no_pad', result_dev['loss_no_pad'], epoch)
            writer.add_scalar('Dev/ppl', result_dev['ppl'], epoch)
            step_nogress += args.eval_epochs
            if args.early_stopping_patient_epochs is not None and step_nogress > args.early_stopping_patient_epochs:
                break

    # testing
    # logger.info(f"Loading best model and compute losses on eval dataset")
    # model.load_state_dict(torch.load('{}'.format(model_ckpt)))
    # result_test = evaluation(model, datahelper, args)
    # log = 'Epoch: {:03d}, Test ppl: {:.4f}  loss: {:.4f}'
    # logger.info(log.format(-1, result_test['ppl'], result_test['loss']))
    logger.info(f"Best model at epoch {best_model_epoch}")
    # writer.add_scalar('Test/nll', result_test['loss'], 0)
    # writer.add_scalar('Test/ppl', result_test['ppl'], 0)


def evaluation(model, dataloader, tokenizer, args):
    # dataset = datahelper.testset if test else datahelper.devset
    # dataset = datahelper.devset
    # data_sampler = SequentialSampler(dataset)
    # dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=args.batch_size,
    #                         collate_fn=dataset.collate_fn)
    model.eval()
    epoch_iterator = tqdm(dataloader, desc="Eval Iteration ")
    eval_loss, eval_loss_no_pad = 0.0, 0.0
    num_steps = 0
    result_dict = {}

    for step, batch in enumerate(epoch_iterator):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(args.device)
            weights = batch["weights"].to(args.device)

            if args.model_type == "bart":
                decoder_input_ids = batch["decoder_input_ids"].to(args.device)
                output = model(input_ids=input_ids,
                               decoder_input_ids=decoder_input_ids,
                               return_dict=True)
                logits = output.logits[:, :-1]
                labels = decoder_input_ids[:, 1:]

            elif args.model_type == "gpt2":
                attention_mask = batch["attention_mask"].to(args.device)
                position_ids = batch.get("position_ids", None)
                if position_ids is not None:
                    position_ids = position_ids.to(args.device)

                output = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               return_dict=True)

                logits = output.logits[:,-batch["max_answer_len"]:-1]
                labels = input_ids[:,-batch["max_answer_len"]+1:]

            loss = batched_decoding_loss(cross_entropy_loss, logits, labels, weights,
                                         pad_token_id=tokenizer.pad_token_id)

        eval_loss += loss.item()
        num_steps += 1
    eval_loss /= num_steps
    # eval_loss_no_pad /= num_steps
    result_dict['loss'] = eval_loss
    result_dict['ppl'] = np.exp(eval_loss)
    return result_dict


def final_evaluation(model, dataloader, tokenizer, args):
    # dataset = datahelper.crowd_devset
    # dataloader = DataLoader(dataset,
    #                         batch_size=args.generation_batch_size,
    #                         shuffle=False,
    #                         collate_fn=dataset.collate_fn)
    answers, verbose_output = generate_answers(model, dataloader, tokenizer, args)

    with open(os.path.join(args.save_dir, "final_answers.jsonl"), "w") as f:
        for ex_dict in verbose_output:
            json_s = json.dumps(ex_dict)
            f.write(json_s + "\n")

    eval_res = evaluate_answers(answers, args)
    with open(os.path.join(args.save_dir, "eval_metrics.pk"), "wb") as f:
        pickle.dump(eval_res, f)


def main():
    parser = configargparse.ArgumentParser(description='Run main.')
    args = get_ansgen_args(parser)
    args.device = torch.device('cuda:{}'.format(args.gpu_device) if torch.cuda.is_available() else 'cpu')
    # ----------------------------------------------------- #
    set_seed(args.seed)
    # model, datahelper, tokenizer = setup_data_and_model(args)
    model, train_dataloader, eval_dataloader, final_dataloader, tokenizer = setup_data_and_model(args)

    if not args.eval_only:
        run_training(model, train_dataloader, eval_dataloader, tokenizer, args)
        logger.info("====== Finished training, now running evaluation ======")

    ckpt_path = os.path.join(args.save_dir, 'model.ckpt')
    model.load_state_dict(torch.load(ckpt_path))
    final_evaluation(model, final_dataloader, tokenizer, args)



if __name__ == '__main__':
    main()