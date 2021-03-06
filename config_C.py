
# FB237
def get_fb237_simple_config(parser):
    parser.add_argument('-bs', '--batch_size', type=int, default=100)
    parser.add_argument('--n_dims_sm', type=int, default=10)
    parser.add_argument('--n_dims', type=int, default=100)
    parser.add_argument('--n_dims_lg', type=int, default=500)
    parser.add_argument('--ent_emb_l2', type=float, default=0.)
    parser.add_argument('--rel_emb_l2', type=float, default=0.)
    parser.add_argument('--query_fn', default='f > g_ng')
    parser.add_argument('--uncon_message_fn', default='f > g_ng')
    parser.add_argument('--uncon_hidden_fn', default='f > g_ng')
    parser.add_argument('--con_message_fn', default='f > g_ng')
    parser.add_argument('--con_hidden_fn', default='f > g_ng')
    parser.add_argument('--transition_fn', default='scorer')
    parser.add_argument('--use_tanh_for_logits', action='store_true', default=False)
    parser.add_argument('--diff_control_for_sampling', nargs=3, type=float, default=[1,2,3])
    parser.add_argument('--diff_control_for_transition', nargs=3, type=float, default=[1,2,3])
    parser.add_argument('--straight_through', action='store_true', default=True)
    parser.add_argument('--norm_mode', default='tanh')
    parser.add_argument('--norm_mix', type=float, default=0.5)
    parser.add_argument('--max_edges_per_example', type=int, default=10000)
    parser.add_argument('--max_attended_nodes', type=int, default=10)
    parser.add_argument('--max_edges_per_node', type=int, default=100)
    parser.add_argument('--max_backtrace_nodes', type=int, default=10)
    parser.add_argument('--backtrace_decay', type=float, default=1.)
    parser.add_argument('--max_seen_nodes', type=int, default=100)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--n_clustering', type=int, default=0)
    parser.add_argument('--n_clusters_per_clustering', type=int, default=0)
    parser.add_argument('--connected_clustering', action='store_true', default=True)
    parser.add_argument('--init_uncon_steps', type=int, default=1)
    parser.add_argument('--simultaneous_uncon_flow', action='store_true', default=False)
    parser.add_argument('--max_steps', type=int, default=5)
    parser.add_argument('--n_rollouts_for_train', type=int, default=1)
    parser.add_argument('--n_rollouts_for_eval', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--dataset', default='FB237')
    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--print_eval', action='store_true', default=True)
    parser.add_argument('--print_eval_freq', type=int, default=100)
    parser.add_argument('--moving_mean_decay', type=float, default=0.99)
    return parser

# FB237
def get_fb237_complex_config(parser):
    parser.add_argument('-bs', '--batch_size', type=int, default=100)
    parser.add_argument('--n_dims_sm', type=int, default=20)
    parser.add_argument('--n_dims', type=int, default=100)
    parser.add_argument('--n_dims_lg', type=int, default=500)
    parser.add_argument('--ent_emb_l2', type=float, default=0.)
    parser.add_argument('--rel_emb_l2', type=float, default=0.)
    parser.add_argument('--query_fn', default='f > g_wg > f_cond > g_wg')
    parser.add_argument('--uncon_message_fn', default='f > g_wg > f_cond > g_wg')
    parser.add_argument('--uncon_hidden_fn', default='f > g_wg > f_cond > g_wg')
    parser.add_argument('--con_message_fn', default='f > g_wg > f_cond > g_wg')
    parser.add_argument('--con_hidden_fn', default='f > g_wg > f_cond > g_wg')
    parser.add_argument('--transition_fn', default='f > g_wg > scorer_cond')
    parser.add_argument('--use_tanh_for_logits', action='store_true', default=False)
    parser.add_argument('--diff_control_for_sampling', nargs=3, type=float, default=[1,2,3])
    parser.add_argument('--diff_control_for_transition', nargs=3, type=float, default=[1,2,3])
    parser.add_argument('--straight_through', action='store_true', default=True)
    parser.add_argument('--norm_mode', default='mix')
    parser.add_argument('--norm_mix', type=float, default=0.5)
    parser.add_argument('--max_edges_per_example', type=int, default=10000)
    parser.add_argument('--max_attended_nodes', type=int, default=10)
    parser.add_argument('--max_edges_per_node', type=int, default=100)
    parser.add_argument('--max_backtrace_nodes', type=int, default=10)
    parser.add_argument('--backtrace_decay', type=float, default=1.)
    parser.add_argument('--max_seen_nodes', type=int, default=100)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--n_clustering', type=int, default=0)
    parser.add_argument('--n_clusters_per_clustering', type=int, default=0)
    parser.add_argument('--connected_clustering', action='store_true', default=True)
    parser.add_argument('--init_uncon_steps', type=int, default=3)
    parser.add_argument('--simultaneous_uncon_flow', action='store_true', default=False)
    parser.add_argument('--max_steps', type=int, default=5)
    parser.add_argument('--n_rollouts_for_train', type=int, default=1)
    parser.add_argument('--n_rollouts_for_eval', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--dataset', default='FB237')
    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--print_eval', action='store_true', default=True)
    parser.add_argument('--print_eval_freq', type=int, default=100)
    parser.add_argument('--moving_mean_decay', type=float, default=0.99)
    return parser

# Toy1
def get_toy1_config(parser):
    parser.add_argument('-bs', '--batch_size', type=int, default=20)
    parser.add_argument('--n_dims_sm', type=int, default=10)
    parser.add_argument('--n_dims', type=int, default=100)
    parser.add_argument('--n_dims_lg', type=int, default=100)
    parser.add_argument('--ent_emb_l2', type=float, default=0.)
    parser.add_argument('--rel_emb_l2', type=float, default=0.)
    parser.add_argument('--query_fn', default='f > g_ng')
    parser.add_argument('--uncon_message_fn', default='f > g_ng')
    parser.add_argument('--uncon_hidden_fn', default='f > g_ng')
    parser.add_argument('--con_message_fn', default='f > g_ng')
    parser.add_argument('--con_hidden_fn', default='f > g_ng')
    parser.add_argument('--transition_fn', default='scorer')
    parser.add_argument('--use_tanh_for_logits', action='store_true', default=False)
    parser.add_argument('--diff_control_for_sampling', nargs=3, type=float, default=[1,2,3])
    parser.add_argument('--diff_control_for_transition', nargs=3, type=float, default=[1,2,3])
    parser.add_argument('--straight_through', action='store_true', default=True)
    parser.add_argument('--norm_mode', default='tanh')
    parser.add_argument('--norm_mix', type=float, default=0.5)
    parser.add_argument('--max_edges_per_example', type=int, default=1000)
    parser.add_argument('--max_attended_nodes', type=int, default=10)
    parser.add_argument('--max_edges_per_node', type=int, default=10)
    parser.add_argument('--max_backtrace_nodes', type=int, default=0)
    parser.add_argument('--backtrace_decay', type=float, default=1.)
    parser.add_argument('--max_seen_nodes', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--n_clustering', type=int, default=0)
    parser.add_argument('--n_clusters_per_clustering', type=int, default=0)
    parser.add_argument('--connected_clustering', action='store_true', default=True)
    parser.add_argument('--init_uncon_steps', type=int, default=2)
    parser.add_argument('--simultaneous_uncon_flow', action='store_true', default=False)
    parser.add_argument('--max_steps', type=int, default=5)
    parser.add_argument('--n_rollouts_for_train', type=int, default=1)
    parser.add_argument('--n_rollouts_for_eval', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dataset', default='Toy1')
    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_metric', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--print_eval', action='store_true', default=False)
    parser.add_argument('--print_eval_freq', type=int, default=100)
    parser.add_argument('--moving_mean_decay', type=float, default=0.99)
    return parser
