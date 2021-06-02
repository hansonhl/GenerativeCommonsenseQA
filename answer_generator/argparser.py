import configargparse

def get_ansgen_args(parser):
    parser.add_argument("-c", "--configs", is_config_file=True)
    parser.add_argument('--grounded_json_dir', type=str)
    parser.add_argument('--graph_info_dir', type=str)
    parser.add_argument('--save_dir', type=str)

    # model
    parser.add_argument('--model_type', type=str, default="gpt2", choices=["gpt2", "bart"])
    # parser.add_argument('--embedding_dim', type=int, default=128)
    # parser.add_argument('--hidden_dim', type=int, default=128)
    # parser.add_argument('--nhead', type=int, default=1)
    # parser.add_argument('--nlayer', type=int, default=1)

    parser.add_argument('--no_concepts', action="store_true")

    # training
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-w', type=float, default=0.0)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_step', type=int, default=50000)
    parser.add_argument('--logging_step', type=int, default=2000)
    parser.add_argument('--early_stopping', action="store_true")

    # gpu option
    parser.add_argument('--gpu_device', type=str, default='0')

    args = parser.parse_args()
    return args

def get_iterative_argparser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--configs", is_config_file=True)
    parser.add_argument("--preprocessed_dir", type=str, required=True)
    parser.add_argument("--graph_info_dir", type=str, required=True)

    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--lm_type", type=str, help="type of language model")
    parser.add_argument("--lm_load_path", type=str, help="Trained language model to further finetune on path generation")
    parser.add_argument("--model_load_path", type=str, help="Trained model to evaluate")
    parser.add_argument("--model_save_name", default="iterative_pathgen.pt", type=str)
    ## My parameters
    parser.add_argument("--evaluate_metrics", default='concept_recall', type=str, help='')

    parser.add_argument("--eval_only", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--generate_with_teacher_forcing", action='store_true', help="Generate answers with teacher forcing")
    parser.add_argument("--generate_only", action='store_true', help="Generate answers without any teacher forcing")
    parser.add_argument("--eval_crowdsourced", action='store_true', help="Generate using crowdsourced dev set")
    parser.add_argument("--graph_expansion_rough_filtering", action='store_true', help="Rough filtering during graph expansion")

    parser.add_argument("--do_decode", action="store_true", help="Whether to do decoding during training")
    parser.add_argument("--num_hops", default=3, type=int)
    parser.add_argument("--filter_concepts_by_routing_score", action="store_true", help="Filter out top_n_routing_concepts then calculate concept loss")
    parser.add_argument("--loss_reweighting", type=str)
    parser.add_argument("--concept_cls_delay_epochs", default=0, type=int)

    parser.add_argument("--max_num_rlns_per_edge", default=3, type=int)
    parser.add_argument("--train_top_n_decoding_concepts", default=8, type=int, help="Select top n concepts to decode")
    parser.add_argument("--train_top_n_routing_concepts", default=100, type=int, help="Select top n concepts to route to on each iteration")
    parser.add_argument("--train_max_num_edges", default=1200, type=int, help="Max number of edges to include when expanding the graph")
    parser.add_argument("--eval_top_n_decoding_concepts", default=8, type=int, help="Select top n concepts to decode")
    parser.add_argument("--eval_top_n_routing_concepts", default=100, type=int, help="Select top n concepts to put into classifier")
    parser.add_argument("--eval_top_n_routing_tails", default=8, type=int, help="Select top n concepts to use as heads in next step")
    parser.add_argument("--eval_max_num_edges", default=1200, type=int, help="Max number of edges to include when expanding the graph")

    parser.add_argument("--concept_loss_weight", default=1.0, type=float)
    parser.add_argument("--edge_loss_weight", default=1.0, type=float)
    parser.add_argument("--decoding_loss_weight", default=1.0, type=float)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)

    parser.add_argument("--learning_rate", default=1e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_ratio", default=0, type=float,
                        help="Linear warmup over warmup_ratio.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--validate_per_epoch', type=int, default=0.)
    parser.add_argument('--validate_steps', type=int, default=3200)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--debug', action='store_true')

    return parser