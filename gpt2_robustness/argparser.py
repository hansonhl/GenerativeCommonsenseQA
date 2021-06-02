import configargparse

def get_ansgen_args(parser):
    parser.add_argument("-c", "--configs", is_config_file=True)
    parser.add_argument('--preprocessed_dir', type=str)
    parser.add_argument('--graph_info_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument("--protoqa_data_path", type=str)
    parser.add_argument("--seed", type=int, default=39)

    # model
    parser.add_argument('--model_type', type=str, default="gpt2", choices=["gpt2", "bart"])
    # parser.add_argument('--embedding_dim', type=int, default=128)
    # parser.add_argument('--hidden_dim', type=int, default=128)
    # parser.add_argument('--nhead', type=int, default=1)
    # parser.add_argument('--nlayer', type=int, default=1)

    parser.add_argument('--eval_only', action="store_true")
    parser.add_argument('--verbose_generation_output', action="store_true")

    parser.add_argument('--ans_concept_input_type', type=str, default="everything")
    parser.add_argument('--formatting', type=str, default="SEP")
    parser.add_argument('--no_permutation_invariant', action="store_true")
    parser.add_argument('--max_additional_concepts', type=int, default=4)
    parser.add_argument('--num_noisy_examples', type=int, default=1)

    parser.add_argument('--multitask_subepochs', type=int, default=20)
    parser.add_argument('--multitask_type', type=str)
    parser.add_argument('--eval_epochs', type=int, default=20)

    # training
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-w', type=float, default=0.0)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--max_step', type=int, default=50000)
    # parser.add_argument('--logging_step', type=int, default=2000)
    parser.add_argument('--early_stopping_patient_epochs', type=int)
    parser.add_argument("--generation_batch_size", type=int, default=16)
    # gpu option
    parser.add_argument('--gpu_device', type=str, default='0')

    args = parser.parse_args()
    return args
