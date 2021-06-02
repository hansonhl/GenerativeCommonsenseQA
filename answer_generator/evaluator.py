import torch
from torch.utils.data import DataLoader
import json
from argparse import ArgumentParser
import transformers
from tqdm import tqdm

from collections import defaultdict
from utils import set_seed, load_graph_info2
from dataset import AnswerGeneratorDataset

from protoqa_evaluator.data_processing import load_question_answer_clusters_from_jsonl
from protoqa_evaluator.evaluation import multiple_evals
from protoqa_evaluator.common_evaluations import exact_match_all_eval_funcs, wordnet_all_eval_funcs

def setup(args):
    if args.model_type == "bart":
        print("loading tokenizer...")
        tokenizer = transformers.BartTokenizer.from_pretrained("facebook/bart-base")
        print("loading model...")
        model = transformers.BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    elif args.model_type == "gpt2":
        print("loading tokenizer...")
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        print("loading model...")
        model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
    else:
        raise  ValueError(f"Invalid model type {args.model_type}")

    print("loading ground truth questions and answer concepts from grounded_jsonl_path")
    with open(args.ground_truth, "r") as f:
        examples = [json.loads(line.strip()) for line in f]

    graph_info = load_graph_info2(args.graph_info_dir)
    dataset = AnswerGeneratorDataset(
        tokenizer, examples, graph_info,
        ans_concept_input_type=args.ground_truth_type,
        model_type=args.model_type, teacher_forcing=False
    )
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.generation_batch_size,
                            collate_fn=dataset.collate_fn)
    model.resize_token_embeddings(len(tokenizer))

    model.load_state_dict(torch.load(args.model_load_path))
    print("loading graph info...")

    return model, tokenizer, graph_info, dataset, dataloader

def ground_truth_concepts(grounded_jsonl_path, tokenizer, graph_info, type):
    print("loading ground truth questions and answer concepts from grounded_jsonl_path")
    with open(grounded_jsonl_path, "r") as f:
        examples = [json.loads(line.strip()) for line in f]

    return AnswerGeneratorDataset(tokenizer, examples, graph_info, type)

def score_by_generator(outputs, scores, tokenizer):
    max_len = min(outputs.shape[1], scores.shape[0])
    scores = scores[:max_len]
    outputs = outputs[:,:max_len]
    indices = outputs.T.unsqueeze(-1)


    scores = scores.gather(dim=2, index=indices)
    scores = scores.squeeze(-1).T
    mask = ~(torch.eq(outputs, tokenizer.pad_token_id)
             | torch.eq(outputs, tokenizer.eos_token_id)
             | torch.eq(outputs, tokenizer.bos_token_id))
    path_lens = mask.sum(dim=1)
    scores[~mask] = 0.
    scores = scores.sum(dim=1).div(path_lens)
    return scores

def decode_answers(outputs, tokenizer):
    return [tokenizer.decode(
        g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for g in outputs]


def generate_answers(model, dataloader, tokenizer, args):
    model = model.to(args.device)

    qid2ans = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generate Answers"):
            if args.model_type == "bart":
                input_ids = batch["input_ids"].to(args.device)
                greedy_output = model.generate(
                    input_ids,
                    max_length=512,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                outputs = greedy_output.sequences
                scores = torch.stack(greedy_output.scores)
            elif args.model_type == "gpt2":
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                position_ids = batch.get("position_ids", None)
                if position_ids is not None:
                    position_ids = position_ids.to(args.device)
                # for i, _ids in enumerate(batch["input_ids"]):
                #     print("---------")
                #     for tok, pos_id, mask  in zip(tokenizer.convert_ids_to_tokens(_ids), position_ids[i], attention_mask[i]):
                #         print(f"{tok}, {pos_id}, {mask}")
                greedy_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    max_length=512,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                input_max_len = input_ids.shape[1]
                outputs = greedy_output.sequences[:,input_max_len:]
                scores = torch.stack(greedy_output.scores)
            else:
                raise ValueError(f"Invalid model type {args.model_type}")

            answer_strs = decode_answers(outputs, tokenizer)
            # for _o, _a in zip(outputs, answer_strs):
            #     print(_o)
            #     print(tokenizer.convert_ids_to_tokens(_o))
            #     print(f"answer: {_a}\n")

            scores = score_by_generator(outputs, scores, tokenizer)
            for qid, answer_str, score in zip(batch["qids"], answer_strs, scores):
                qid2ans[qid].append((score.item(), answer_str))
            # break

    res = {}
    for qid, answers in qid2ans.items():
        ranked_answers = [a for (_, a) in sorted(answers, reverse=True)]
        res[qid] = ranked_answers

    return res

def evaluate_answers(answers, args):
    question_data = load_question_answer_clusters_from_jsonl(args.protoqa_data_path)
    print("\n----------Exact Match----------")
    multiple_evals(
        eval_func_dict=exact_match_all_eval_funcs,
        question_data=question_data,
        answers_dict=answers,
    )
    print("\n------------Wordnet-----------")
    multiple_evals(
        eval_func_dict=wordnet_all_eval_funcs,
        question_data=question_data,
        answers_dict=answers,
    )



def main():
    parser = ArgumentParser()
    parser.add_argument("model_load_path", type=str)
    parser.add_argument("--model_type", type=str, default="gpt2")
    parser.add_argument("--protoqa_data_path", type=str)
    parser.add_argument("--graph_info_dir", type=str)
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument("--ground_truth", type=str)
    parser.add_argument("--ground_truth_type", type=str)
    parser.add_argument("--gpu_device", type=str, default='0')
    parser.add_argument("--generation_batch_size", type=int, default=128)
    # parser.add_argument("")
    args = parser.parse_args()

    args.device = torch.device(f'cuda:{args.gpu_device}' if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed)

    model, tokenizer, graph_info, dataset, dataloader = setup(args)
    
    # if args.ground_truth:
    #     dataset = ground_truth_concepts(args.ground_truth, tokenizer, graph_info, args.ground_truth_type)
    # else:
    #     raise NotImplementedError
    
    answers = generate_answers(model, dataloader, tokenizer, args)
    evaluate_answers(answers, args)

if __name__ == '__main__':
    pass