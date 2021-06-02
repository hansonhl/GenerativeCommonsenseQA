import configargparse
import os
import numpy as np
import logging
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import transformers

from argparser import get_ansgen_args
from dataset import AnsgenTrainingDatahelper
from utils import batched_decoding_loss

torch.set_num_threads(4)

logger = logging.getLogger()

cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")


def run_training(args):
    # ----------------------------------------------------- #
    # checkpoint directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_ckpt = os.path.join(args.save_dir, 'model.ckpt')

    # log file
    if args.num_epoch == 0:
        log_path = os.path.join(args.save_dir, 'test.log')
    else:
        log_path = os.path.join(args.save_dir, 'train.log')

    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path, 'w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info('args: {}'.format(args))

    writer = SummaryWriter(log_dir=args.save_dir)
    # ----------------------------------------------------- #
    # load data & init model and optimizer

    logger.info('Loading data & model')

    answer_concept_type = "combination" if not args.no_concepts else "no_concepts"
    if args.model_type == "bart":
        tokenizer = transformers.BartTokenizer.from_pretrained("facebook/bart-base")
        datahelper = AnsgenTrainingDatahelper(
            tokenizer, args.grounded_json_dir, args.graph_info_dir,
            answer_concepts=answer_concept_type, model_type=args.model_type, teacher_forcing=True
        )
        model = transformers.BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    elif args.model_type == "gpt2":
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        datahelper = AnsgenTrainingDatahelper(
            tokenizer, args.grounded_json_dir, args.graph_info_dir,
            answer_concepts=answer_concept_type, model_type=args.model_type, teacher_forcing=True)
        model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
        model.resize_token_embeddings(len(tokenizer))
    else:
        raise ValueError(f"Invalid model type {args.model_type}")

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

    train_sampler = RandomSampler(datahelper.trainset)
    train_dataloader = DataLoader(datahelper.trainset, sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=datahelper.trainset.collate_fn)
    logger.info('Num of samples: {}, steps: {}'.format(len(datahelper.trainset), len(datahelper.trainset)//args.batch_size))

    t_total = len(train_dataloader) * args.num_epoch
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #     ]
    optimizer = transformers.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    # ----------------------------------------------------- #

    # training
    best_dev_loss = 1e19
    train_iterator = trange(int(args.num_epoch), desc="Epoch")
    step_nogress = 0
    global_step = 0
    save_id = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_model_epoch = 0
    for epoch in train_iterator:
        train_loss = 0.0
        num_steps = 0
        model.train()
        epoch_iterator = tqdm(train_dataloader, desc="Train Iteration at Epoch {}".format(epoch))
        for step, batch in enumerate(epoch_iterator):
            optimizer.zero_grad()
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
            # writer.add_scalar('Train/nll_no_pad', loss_no_pad.item(), global_step)

            global_step += 1

        train_loss /= num_steps
        log = 'Epoch: {:03d} Train loss: {:.4f}'
        logger.info(log.format(epoch, train_loss))

        result_dev = evaluation(datahelper, model, args)
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
        step_nogress += 1
        if step_nogress > 2 and args.early_stopping:
            break

    # testing
    model.load_state_dict(torch.load('{}'.format(model_ckpt)))
    result_test = evaluation(datahelper, model, args)
    log = 'Epoch: {:03d}, Test ppl: {:.4f}  loss: {:.4f}'
    logger.info(log.format(-1, result_test['ppl'], result_test['loss']))
    logger.info(f"Best model at epoch {best_model_epoch}")
    writer.add_scalar('Test/nll', result_test['loss'], 0)
    # writer.add_scalar('Test/nll_no_pad', result_test['loss_no_pad'], 0)
    writer.add_scalar('Test/ppl', result_test['ppl'], 0)

def evaluation(datahelper, model, args):
    # dataset = datahelper.testset if test else datahelper.devset
    dataset = datahelper.devset
    data_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=args.batch_size,
                            collate_fn=dataset.collate_fn)
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
                                         pad_token_id=dataset.tokenizer.pad_token_id)

        eval_loss += loss.item()
        num_steps += 1
    eval_loss /= num_steps
    # eval_loss_no_pad /= num_steps
    result_dict['loss'] = eval_loss
    result_dict['ppl'] = np.exp(eval_loss)
    return result_dict

def main():
    parser = configargparse.ArgumentParser(description='Run main.')
    args = get_ansgen_args(parser)
    args.device = torch.device('cuda:{}'.format(args.gpu_device) if torch.cuda.is_available() else 'cpu')
    # ----------------------------------------------------- #

    run_training(args)

if __name__ == '__main__':
    main()