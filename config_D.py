
# FB237
def get_fb237_config(parser):
    parser.add_argument('-bs', '--batch_size', type=int, default=500)
    parser.add_argument('--n_dims_sm', type=int, default=10)
    parser.add_argument('--n_dims', type=int, default=100)
    parser.add_argument('--n_dims_lg', type=int, default=200)
    parser.add_argument('--ent_emb_l2', type=float, default=0.)
    parser.add_argument('--rel_emb_l2', type=float, default=0.)
    parser.add_argument('--use_ent_emb', action='store_true', default=False)

    parser.add_argument('--use_gumbel_softmax', action='store_true', default=False)
    # Options: 'by_edge', 'by_rel'
    parser.add_argument('--probability_mode', default='by_rel')
    parser.add_argument('--temperature', type=float, default=1.)

    parser.add_argument('--max_edges_per_example', type=int, default=10000)
    parser.add_argument('--max_edges_per_node', type=int, default=100)
    parser.add_argument('--max_attended_nodes', type=int, default=10)
    parser.add_argument('--max_backtrace_nodes', type=int, default=0)  # no use
    parser.add_argument('--backtrace_decay', type=float, default=1.)  # no use
    parser.add_argument('--max_seen_nodes', type=int, default=100)

    #parser.add_argument('--diff_control_for_sampling', nargs=3, type=float, default=[1, 2., 3.])
    #parser.add_argument('--diff_control_for_transition', nargs=3, type=float, default=[1., 2., 3.])
    parser.add_argument('--diff_control_for_sampling', default=None)
    parser.add_argument('--diff_control_for_transition', default=None)

    parser.add_argument('--use_gate', action='store_true', default=False)
    # Options: None, 'square', 'tanh'
    parser.add_argument('--uncon_message_cond_mode', default=None)
    parser.add_argument('--uncon_hidden_cond_mode', default='tanh')
    parser.add_argument('--con_message_cond_mode', default=None)
    parser.add_argument('--con_hidden_cond_mode', default='tanh')

    parser.add_argument('--straight_through_trans', action='store_true', default=False)
    parser.add_argument('--straight_through_node', action='store_true', default=False)
    parser.add_argument('--direct_intervention_trans', action='store_true', default=False)
    parser.add_argument('--direct_intervention_node', action='store_true', default=False)
    # Options: None, 'multiply', 'scaled', 'tanh'
    parser.add_argument('--direct_intervention_trans_mode', default='tanh')
    parser.add_argument('--direct_intervention_node_mode', default='tanh')

    # fn(hidden_vi, rel_emb, hidden_vj)
    # Option 1: [[0], [0, 1], [0, 2], [0, 1, 2]])
    # Option 2: [[0, 1, 2]]
    parser.add_argument('--uncon_message_interact', default=[[0, 1, 2]])

    # fn(message_aggr, hidden, ent_emb)
    # Option 1: [[0], [0, 1], [0, 2], [0, 1, 2]]
    # Option 2: [[0, 1, 2]]
    # Option 3: [[0, 1]] (for `use_ent_emb`=False)
    parser.add_argument('--uncon_hidden_interact', default=[[0, 1]])

    # fn(hidden_vi, rel_emb, hidden_vj, query_head_emb, query_rel_emb)
    # Option 1: ([[0], [0, 1], [0, 2], [0, 1, 2]],  [[3, 5], [4, 5], [3, 4, 5]])
    # Option 2: ([[0, 1, 2, 3, 4]],  None)
    # Option 3: use query_head_hidden instead of query_head_emb
    parser.add_argument('--con_message_interacts', default=([[0, 1, 2, 3, 4]],  None))

    # fn(message_aggr, hidden, hidden_uncon, query_head_emb, query_rel_emb)
    # Option 1: ([[0], [0, 1], [0, 2], [0, 1, 2]],  [[3, 5], [4, 5], [3, 4, 5]])
    # Option 2: [[0, 1, 2, 3, 4]],  None)
    # Option 3: use query_head_hidden instead of query_head_emb
    parser.add_argument('--con_hidden_interacts', default=([[0, 1, 2, 3, 4]],
                                                           None))

    # fn(hidden_con_vi, rel_emb, hidden_con_vj, hidden_uncon_vj, query_head_emb, query_rel_emb)
    # Option 1: [[[0, 4, 5], [2, 4, 5]],  [[0, 4, 5], [3, 4, 5]],  [[0, 1, 4, 5], [2, 1, 4, 5]],  [[0, 1, 4, 5], [3, 1, 4, 5]]])
    # Option 2: [[[0, 1, 4, 5], [2, 3, 1, 4, 5]]]
    # Option 3: use query_head_hidden instead of query_head_emb
    parser.add_argument('--att_transition_interact', default=[[[0, 1, 4, 5], [2, 3, 1, 4, 5]]])

    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--n_clustering', type=int, default=0)  # no use
    parser.add_argument('--n_clusters_per_clustering', type=int, default=0)  # no use
    parser.add_argument('--connected_clustering', action='store_true', default=True)  # no use

    parser.add_argument('--init_uncon_steps', type=int, default=2)
    parser.add_argument('--simultaneous_uncon_flow', action='store_true', default=False)  # no use
    parser.add_argument('--max_steps', type=int, default=8)
    parser.add_argument('--n_rollouts_for_train', type=int, default=1)
    parser.add_argument('--n_rollouts_for_eval', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--dataset', default='FB237')
    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_metric', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--print_eval', action='store_true', default=True)
    parser.add_argument('--print_eval_metric', action='store_true', default=True)
    parser.add_argument('--print_eval_freq', type=int, default=100)
    parser.add_argument('--moving_mean_decay', type=float, default=0.99)
    return parser


# Toy1
def get_toy1_config(parser):
    parser.add_argument('-bs', '--batch_size', type=int, default=40)
    parser.add_argument('--n_dims_sm', type=int, default=10)
    parser.add_argument('--n_dims', type=int, default=100)
    parser.add_argument('--n_dims_lg', type=int, default=200)
    parser.add_argument('--ent_emb_l2', type=float, default=0.)
    parser.add_argument('--rel_emb_l2', type=float, default=0.)
    parser.add_argument('--diff_control_for_sampling', nargs=3, type=float, default=[1,2,3])
    parser.add_argument('--diff_control_for_transition', nargs=3, type=float, default=[1,2,3])
    parser.add_argument('--straight_through', action='store_true', default=True)
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
    parser.add_argument('--init_uncon_steps', type=int, default=5)
    parser.add_argument('--simultaneous_uncon_flow', action='store_true', default=False)
    parser.add_argument('--max_steps', type=int, default=5)
    parser.add_argument('--n_rollouts_for_train', type=int, default=1)
    parser.add_argument('--n_rollouts_for_eval', type=int, default=1)
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

# Countries
def get_countries_config(parser):
    parser.add_argument('-bs', '--batch_size', type=int, default=40)
    parser.add_argument('--n_dims_sm', type=int, default=10)
    parser.add_argument('--n_dims', type=int, default=100)
    parser.add_argument('--n_dims_lg', type=int, default=200)
    parser.add_argument('--ent_emb_l2', type=float, default=0.)
    parser.add_argument('--rel_emb_l2', type=float, default=0.)
    parser.add_argument('--diff_control_for_sampling', nargs=3, type=float, default=[1,2,3])
    parser.add_argument('--diff_control_for_transition', nargs=3, type=float, default=[1,2,3])
    parser.add_argument('--straight_through', action='store_true', default=False)
    parser.add_argument('--max_edges_per_example', type=int, default=1000)
    parser.add_argument('--max_attended_nodes', type=int, default=10)
    parser.add_argument('--max_edges_per_node', type=int, default=10)
    parser.add_argument('--max_backtrace_nodes', type=int, default=0)  # no use
    parser.add_argument('--backtrace_decay', type=float, default=1.)  # no use
    parser.add_argument('--max_seen_nodes', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--n_clustering', type=int, default=0)  # no use
    parser.add_argument('--n_clusters_per_clustering', type=int, default=0)  # no use
    parser.add_argument('--connected_clustering', action='store_true', default=True)  # no use
    parser.add_argument('--init_uncon_steps', type=int, default=5)
    parser.add_argument('--simultaneous_uncon_flow', action='store_true', default=False)  # no use
    parser.add_argument('--max_steps', type=int, default=5)
    parser.add_argument('--n_rollouts_for_train', type=int, default=1)
    parser.add_argument('--n_rollouts_for_eval', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dataset', default='Countries')
    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_metric', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--print_eval', action='store_true', default=False)
    parser.add_argument('--print_eval_freq', type=int, default=100)
    parser.add_argument('--moving_mean_decay', type=float, default=0.99)
    return parser