# GenerativeCommonsenseQA

This is the main repo for the final project for Stanford CS229, Spring 2021.

## Requirements

```
python >= 3.8
pytorch >= 1.8.1
transformers >= 4.2.2
networkx
```

## Main experiments

`answer_generator/` contains all the dataset and model implementation and training for the concept retriever-answer generator
architecture. To train/eval it, run 

```
$ python train_iterative.py -c CONFIG_FILE --save_dir SAVE_DIR
```

where `CONFIG_FILE` is one of the config files in the `answer_generator/configs/`
directory. Some config files are for training while some are for evaluating.

`SAVE_DIR` is a directory to save the model checkpoints and 
generation outputs of the model.


## Answer generator robustness analysis

`gpt2_robustness/` contains all the dataset implementation, as well as
the training script for analyzing the robustness of the GPT-2-based answer 
generator to noisy concepts. In this case we directly use an off-the shelf
pre-trained GPT-2 model from the `transformers` library. 

To run the training, run
```
$ python train_iterative.py -c CONFIG_FILE --save_dir SAVE_DIR
```
Similar to the above, `CONFIG_FILE` is one of the config files in 
`gpt2_robustness/configs/`, and `SAVE_DIR` is a directory of your choice to 
save the model checkpoints and outputs. This training script will automatically
do evaluation at the end of training.

## Data

Please contact the author of the project for the data.

The contents of this repo are refactored out of a large repo that is very messy,
so there is no commit history.
