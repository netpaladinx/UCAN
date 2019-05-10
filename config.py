
# FB237
def get_fb237_config(parser):
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_dims_sm', type=int, default=50)
    parser.add_argument('--n_dims', type=int, default=100)

    parser.add_argument('--max_edges_per_example', type=int, default=10000)
    parser.add_argument('--max_edges_per_node', type=int, default=200)
    parser.add_argument('--max_attended_nodes', type=int, default=10)
    parser.add_argument('--max_seen_nodes', type=int, default=200)
    parser.add_argument('--drop_edges_per_node', type=float, default=None)

    parser.add_argument('--test_batch_size', type=int, default=50)
    parser.add_argument('--test_max_edges_per_example', type=int, default=10000)
    parser.add_argument('--test_max_edges_per_node', type=int, default=400)
    parser.add_argument('--test_max_attended_nodes', type=int, default=20)
    parser.add_argument('--test_max_seen_nodes', type=int, default=400)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--aggregate_op', default='mean_v3')
    parser.add_argument('--uncon_steps', type=int, default=4)
    parser.add_argument('--con_steps', type=int, default=8)

    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipnorm', type=float, default=1.)

    parser.add_argument('--dataset', default='FB237')
    parser.add_argument('--remove_all_head_tail_edges', action='store_true', default=True)

    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_metric', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--eval_within_epoch', default=[])
    parser.add_argument('--print_eval', action='store_true', default=True)
    parser.add_argument('--print_eval_metric', action='store_true', default=True)
    parser.add_argument('--print_eval_freq', type=int, default=100)
    parser.add_argument('--moving_mean_decay', type=float, default=0.99)

    return parser


# FB15K
def get_fb15k_config(parser):
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_dims_sm', type=int, default=50)
    parser.add_argument('--n_dims', type=int, default=100)

    parser.add_argument('--max_edges_per_example', type=int, default=10000)
    parser.add_argument('--max_edges_per_node', type=int, default=150)
    parser.add_argument('--max_attended_nodes', type=int, default=10)
    parser.add_argument('--max_seen_nodes', type=int, default=150)
    parser.add_argument('--drop_edges_per_node', type=float, default=None)

    parser.add_argument('--test_batch_size', type=int, default=50)
    parser.add_argument('--test_max_edges_per_example', type=int, default=10000)
    parser.add_argument('--test_max_edges_per_node', type=int, default=300)
    parser.add_argument('--test_max_attended_nodes', type=int, default=20)
    parser.add_argument('--test_max_seen_nodes', type=int, default=300)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--aggregate_op', default='mean_v3')
    parser.add_argument('--uncon_steps', type=int, default=4)
    parser.add_argument('--con_steps', type=int, default=8)

    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipnorm', type=float, default=5.)

    parser.add_argument('--dataset', default='FB15K')
    parser.add_argument('--remove_all_head_tail_edges', action='store_true', default=False)

    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_metric', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--eval_within_epoch', default=[])
    parser.add_argument('--print_eval', action='store_true', default=True)
    parser.add_argument('--print_eval_metric', action='store_true', default=True)
    parser.add_argument('--print_eval_freq', type=int, default=100)
    parser.add_argument('--moving_mean_decay', type=float, default=0.99)

    return parser


# WN18RR
def get_wn18rr_config(parser):
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_dims_sm', type=int, default=50)
    parser.add_argument('--n_dims', type=int, default=100)

    parser.add_argument('--max_edges_per_example', type=int, default=10000)
    parser.add_argument('--max_edges_per_node', type=int, default=400)
    parser.add_argument('--max_attended_nodes', type=int, default=20)
    parser.add_argument('--max_seen_nodes', type=int, default=400)
    parser.add_argument('--drop_edges_per_node', type=float, default=None)

    parser.add_argument('--test_batch_size', type=int, default=50)
    parser.add_argument('--test_max_edges_per_example', type=int, default=10000)
    parser.add_argument('--test_max_edges_per_node', type=int, default=800)
    parser.add_argument('--test_max_attended_nodes', type=int, default=40)
    parser.add_argument('--test_max_seen_nodes', type=int, default=800)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--aggregate_op', default='mean_v3')
    parser.add_argument('--uncon_steps', type=int, default=4)
    parser.add_argument('--con_steps', type=int, default=8)

    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipnorm', type=float, default=1.)

    parser.add_argument('--dataset', default='WN18RR')
    parser.add_argument('--remove_all_head_tail_edges', action='store_true', default=False)

    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_metric', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--eval_within_epoch', default=[])
    parser.add_argument('--print_eval', action='store_true', default=True)
    parser.add_argument('--print_eval_metric', action='store_true', default=True)
    parser.add_argument('--print_eval_freq', type=int, default=100)
    parser.add_argument('--moving_mean_decay', type=float, default=0.99)

    return parser


# WN
def get_wn_config(parser):
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_dims_sm', type=int, default=50)
    parser.add_argument('--n_dims', type=int, default=100)

    parser.add_argument('--max_edges_per_example', type=int, default=10000)
    parser.add_argument('--max_edges_per_node', type=int, default=300)
    parser.add_argument('--max_attended_nodes', type=int, default=20)
    parser.add_argument('--max_seen_nodes', type=int, default=300)
    parser.add_argument('--drop_edges_per_node', type=float, default=None)

    parser.add_argument('--test_batch_size', type=int, default=50)
    parser.add_argument('--test_max_edges_per_example', type=int, default=10000)
    parser.add_argument('--test_max_edges_per_node', type=int, default=600)
    parser.add_argument('--test_max_attended_nodes', type=int, default=40)
    parser.add_argument('--test_max_seen_nodes', type=int, default=600)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--aggregate_op', default='mean_v3')
    parser.add_argument('--uncon_steps', type=int, default=4)
    parser.add_argument('--con_steps', type=int, default=8)

    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipnorm', type=float, default=5.)

    parser.add_argument('--dataset', default='WN')
    parser.add_argument('--remove_all_head_tail_edges', action='store_true', default=False)

    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_metric', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--eval_within_epoch', default=[])
    parser.add_argument('--print_eval', action='store_true', default=True)
    parser.add_argument('--print_eval_metric', action='store_true', default=True)
    parser.add_argument('--print_eval_freq', type=int, default=100)
    parser.add_argument('--moving_mean_decay', type=float, default=0.99)

    return parser


# YAGO310
def get_yago310_config(parser):
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_dims_sm', type=int, default=50)
    parser.add_argument('--n_dims', type=int, default=100)

    parser.add_argument('--max_edges_per_example', type=int, default=10000)
    parser.add_argument('--max_edges_per_node', type=int, default=200)
    parser.add_argument('--max_attended_nodes', type=int, default=10)
    parser.add_argument('--max_seen_nodes', type=int, default=200)
    parser.add_argument('--drop_edges_per_node', type=float, default=None)

    parser.add_argument('--test_batch_size', type=int, default=50)
    parser.add_argument('--test_max_edges_per_example', type=int, default=10000)
    parser.add_argument('--test_max_edges_per_node', type=int, default=400)
    parser.add_argument('--test_max_attended_nodes', type=int, default=20)
    parser.add_argument('--test_max_seen_nodes', type=int, default=400)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--aggregate_op', default='mean_v3')
    parser.add_argument('--uncon_steps', type=int, default=3)
    parser.add_argument('--con_steps', type=int, default=6)

    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipnorm', type=float, default=1.)

    parser.add_argument('--dataset', default='YAGO310')
    parser.add_argument('--remove_all_head_tail_edges', action='store_true', default=False)

    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_metric', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--eval_within_epoch', default=[])
    parser.add_argument('--print_eval', action='store_true', default=True)
    parser.add_argument('--print_eval_metric', action='store_true', default=True)
    parser.add_argument('--print_eval_freq', type=int, default=100)
    parser.add_argument('--moving_mean_decay', type=float, default=0.99)

    return parser


# Nell995: for separate learning per subset
def get_nell995_separate_config(parser):
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--n_dims_sm', type=int, default=50)
    parser.add_argument('--n_dims', type=int, default=100)

    parser.add_argument('--max_edges_per_example', type=int, default=20000)
    parser.add_argument('--max_edges_per_node', type=int, default=200)
    parser.add_argument('--max_attended_nodes', type=int, default=100)
    parser.add_argument('--max_seen_nodes', type=int, default=200)
    parser.add_argument('--drop_edges_per_node', type=float, default=None)

    parser.add_argument('--test_batch_size', type=int, default=20)
    parser.add_argument('--test_max_edges_per_example', type=int, default=20000)
    parser.add_argument('--test_max_edges_per_node', type=int, default=200)
    parser.add_argument('--test_max_attended_nodes', type=int, default=100)
    parser.add_argument('--test_max_seen_nodes', type=int, default=200)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--aggregate_op', default='mean_v3')
    parser.add_argument('--uncon_steps', type=int, default=4)
    parser.add_argument('--con_steps', type=int, default=8)

    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipnorm', type=float, default=1.)

    parser.add_argument('--dataset', default='Nell995')
    parser.add_argument('--remove_all_head_tail_edges', action='store_true', default=False)

    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_metric', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--eval_within_epoch', default=[])
    parser.add_argument('--print_eval', action='store_true', default=True)
    parser.add_argument('--print_eval_metric', action='store_true', default=True)
    parser.add_argument('--print_eval_freq', type=int, default=100)
    parser.add_argument('--moving_mean_decay', type=float, default=0.9)

    return parser


# Nell995: for joint learning over subsets
def get_nell995_joint_config(parser):
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--n_dims_sm', type=int, default=50)
    parser.add_argument('--n_dims', type=int, default=100)

    parser.add_argument('--max_edges_per_example', type=int, default=40000)
    parser.add_argument('--max_edges_per_node', type=int, default=200)
    parser.add_argument('--max_attended_nodes', type=int, default=400)
    parser.add_argument('--max_seen_nodes', type=int, default=800)
    parser.add_argument('--drop_edges_per_node', type=float, default=None)

    parser.add_argument('--test_batch_size', type=int, default=20)
    parser.add_argument('--test_max_edges_per_example', type=int, default=40000)
    parser.add_argument('--test_max_edges_per_node', type=int, default=200)
    parser.add_argument('--test_max_attended_nodes', type=int, default=400)
    parser.add_argument('--test_max_seen_nodes', type=int, default=800)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--aggregate_op', default='mean_v3')
    parser.add_argument('--uncon_steps', type=int, default=2)
    parser.add_argument('--con_steps', type=int, default=4)

    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipnorm', type=float, default=1.)

    parser.add_argument('--dataset', default='Nell995_v2')
    parser.add_argument('--remove_all_head_tail_edges', action='store_true', default=False)

    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_metric', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--moving_mean_decay', type=float, default=0.9)

    return parser