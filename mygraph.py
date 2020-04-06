"""Graph utilities."""

# from time import time
import openne.graph
import networkx as nx

class Graph_Int(openne.graph.Graph):
    def __init__(self):
        super().__init__()

    def encode_node(self):
        super().encode_node()

    def read_g(self, g):
        super().read_g(g)

    def read_adjlist(self, filename):
        """ Read graph from adjacency file in which the edge must be unweighted
            the format of each line: v1 n1 n2 n3 ... nk
            :param filename: the filename of input file
        """
        super().read_adjlist(filename)

    def read_edgelist(self, filename, node_index, weighted=False, directed=False):
        self.G = nx.DiGraph()

        if directed:
            def read_unweighted(l):
                _src, _dst = [x for x in l.split()]
                src = node_index[int(_src)]
                dst = node_index[int(_dst)]
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = 1.0

            def read_weighted(l):
                _src, _dst, w = [x for x in l.split()]
                src = node_index[int(_src)]
                dst = node_index[int(_dst)]
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = float(w)
        else:
            def read_unweighted(l):
                _src, _dst = [x for x in l.split()]
                src = node_index[int(_src)]
                dst = node_index[int(_dst)]
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = 1.0
                self.G[dst][src]['weight'] = 1.0

            def read_weighted(l):
                _src, _dst, w = [x for x in l.split()]
                src = node_index[int(_src)]
                dst = node_index[int(_dst)]
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = float(w)
                self.G[dst][src]['weight'] = float(w)
        fin = open(filename, 'r')
        func = read_unweighted
        if weighted:
            func = read_weighted
        while 1:
            l = fin.readline()
            if l == '':
                break
            func(l)
        fin.close()
        self.encode_node()

    def read_node_label(self, filename):
        super().read_node_label(filename)

    def read_node_features(self, filename):
        super().read_node_features(filename)

    def read_node_status(self, filename):
        super().read_node_status(filename)

    def read_edge_label(self, filename):
        super().read_edge_label(filename)


class Graph_Str(openne.graph.Graph):
    def __init__(self):
        super().__init__()

    def encode_node(self):
        super().encode_node()

    def read_g(self, g):
        super().read_g(g)

    def read_adjlist(self, filename):
        """ Read graph from adjacency file in which the edge must be unweighted
            the format of each line: v1 n1 n2 n3 ... nk
            :param filename: the filename of input file
        """
        super().read_adjlist(filename)

    def read_edgelist(self, filename, node_index, weighted=False, directed=False):
        self.G = nx.DiGraph()

        if directed:
            def read_unweighted(l):
                _src, _dst = [x for x in l.split()]
                src = str(node_index[int(_src)])
                dst = str(node_index[int(_dst)])
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = 1.0

            def read_weighted(l):
                _src, _dst, w = [x for x in l.split()]
                src = str(node_index[int(_src)])
                dst = str(node_index[int(_dst)])
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = float(w)
        else:
            def read_unweighted(l):
                _src, _dst = [x for x in l.split()]
                src = str(node_index[int(_src)])
                dst = str(node_index[int(_dst)])
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = 1.0
                self.G[dst][src]['weight'] = 1.0

            def read_weighted(l):
                _src, _dst, w = [x for x in l.split()]
                src = str(node_index[int(_src)])
                dst = str(node_index[int(_dst)])
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = float(w)
                self.G[dst][src]['weight'] = float(w)
        fin = open(filename, 'r')
        func = read_unweighted
        if weighted:
            func = read_weighted
        while 1:
            l = fin.readline()
            if l == '':
                break
            func(l)
        fin.close()
        self.encode_node()

    def read_node_label(self, filename):
        super().read_node_label(filename)

    def read_node_features(self, filename):
        super().read_node_features(filename)

    def read_node_status(self, filename):
        super().read_node_status(filename)

    def read_edge_label(self, filename):
        super().read_edge_label(filename)