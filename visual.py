import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import networkx as nx
import numpy as np

class Visualizer(object):
    def _get_node_params_at_step(self, t, nodes, size=300, inflate=1., min_max=0.1):
        node_atts = [n[1]['node_att'][t] for n in nodes]
        cols = node_atts
        c = np.array(cols)
        vmin = c.min()
        vmax = max(c.max(), min_max)
        sizes = [size * (1+a*inflate) for a in node_atts]
        return cols, vmin, vmax, sizes

    def _get_edge_params_at_step(self, t, edges, width=1., inflate=2., min_max=0.1):
        if t == 0:
            cols = [0.] * len(edges)
            widths = [1.] * len(edges)
            return cols, 0., min_max, widths
        else:
            edge_atts = [e[2]['trans_att'][t-1] for e in edges]
            cols = edge_atts
            c = np.array(cols)
            vmin = c.min()
            vmax = c.max()
            widths = [width * (1+a*inflate) for a in edge_atts]
            return cols, vmin, vmax, widths

    def _get_node_labels_at_step(self, t, nodes, att_level=0.5):
        labels_unatt = dict((n[0], n[0]) for n in nodes if n[1]['node_att'][t] < att_level)
        labels_att = dict((n[0], n[0]) for n in nodes if n[1]['node_att'][t] >= att_level)
        fcols_unatt = 'w'
        fcols_att = 'k'
        return labels_unatt, fcols_unatt, labels_att, fcols_att

    def draw(self, nodes, edges):
        fig, ax = plt.subplots(figsize=(6,4))

        graph = nx.MultiDiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        #pos = nx.nx_agraph.graphviz_layout(graph, prog='twopi', root=0)
        pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')
        #pos = nx.spring_layout(graph)

        T = len(nodes[0][1]['node_att'])

        cdict = {'red': [(0.0, 0.4, 0.4),
                         (1.0, 1.0, 1.0)],

                 'green': [(0.0, 0.4, 0.4),
                           (1.0, 0.8, 0.8)],

                 'blue': [(0.0, 0.4, 0.4),
                          (1.0, 0.0, 0.0)]}

        attcmp = colors.LinearSegmentedColormap('attCmap', segmentdata=cdict, N=256)

        def update(t):
            ax.clear()

            cols, vmin, vmax, sizes = self._get_node_params_at_step(t, nodes)
            nx.draw_networkx_nodes(graph, pos, node_color=cols, vmin=vmin, vmax=vmax, cmap=attcmp, node_size=sizes,
                                   edgecolors='k', ax=ax)

            e_cols, e_vmin, e_vmax, widths = self._get_edge_params_at_step(t, edges)
            nx.draw_networkx_edges(graph, pos, width=widths, edge_color=e_cols, edge_vmin=e_vmin, edge_vmax=e_vmax,
                                   edge_cmap=attcmp, arrowstyle='->', ax=ax)

            lab_ua, fc_ua, lab_a, fc_a = self._get_node_labels_at_step(t, nodes, att_level=vmax*0.5)
            nx.draw_networkx_labels(graph, pos, labels=lab_ua, font_color=fc_ua, ax=ax)
            nx.draw_networkx_labels(graph, pos, labels=lab_a, font_color=fc_a, ax=ax)

            ax.set_title("Step %d" % (t + 1), fontweight="bold")

        ani = animation.FuncAnimation(fig, update, frames=3, interval=1000, repeat=True)
        ani.save('ani.gif', writer='imagemagick', bitrate=300)
        plt.show()


if __name__ == '__main__':
    nodes = [(0, {'node_att': [1.0, 0.3, 0.0], 'is_source': True}),
             (1, {'node_att': [0.0, 0.0, 0.0]}),
             (2, {'node_att': [0.0, 0.4, 0.2]}),
             (3, {'node_att': [0.0, 0.3, 0.1]}),
             (4, {'node_att': [0.0, 0.0, 0.1]}),
             (5, {'node_att': [0.0, 0.0, 0.6], 'is_target': True})]

    edges = [(0, 2, {'trans_att': [0.4, 0.2]}),
             (0, 3, {'trans_att': [0.3, 0.1]}),
             (1, 2, {'trans_att': [0.0, 0.0]}),
             (1, 3, {'trans_att': [0.0, 0.0]}),
             (2, 4, {'trans_att': [0.0, 0.1]}),
             (2, 5, {'trans_att': [0.0, 0.3]}),
             (3, 4, {'trans_att': [0.0, 0.0]}),
             (3, 5, {'trans_att': [0.0, 0.3]})]

    vis = Visualizer()
    vis.draw(nodes, edges)
