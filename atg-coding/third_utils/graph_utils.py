from collections import defaultdict, namedtuple, OrderedDict
from copy import deepcopy
from queue import Queue


def topology_sort(nodes, src2dsts):
    def cal_in_degree():
        in_degree = {}
        for node in nodes:
            in_degree[node] = 0
        for src, dsts in src2dsts.items():
            for dst in dsts:
                in_degree[dst] += 1
        return in_degree

    # calculate in degree of each node
    in_degree = cal_in_degree()
    # get zero in degree nodes
    zero_in_degree_nodes = [node for node, degree in in_degree.items() if degree == 0]
    results = []
    while len(zero_in_degree_nodes) != 0:
        # delete zero in degree node, and if generate new zero in degree node, add to zero in degree nodes stack.
        zero_node = zero_in_degree_nodes.pop()
        results.append(zero_node)
        if zero_node in src2dsts:  # have dst node with zero_node as src
            for dst in src2dsts[zero_node]:
                in_degree[dst] -= 1
                if in_degree[dst] == 0:
                    zero_in_degree_nodes.append(dst)
    assert len(results) == len(in_degree)
    return results


def traversal_source_nodes(node, dst2srcs, prior='depth', solver=lambda node: node):
    def traversal_sources_depth_first(node, dst2srcs, solver, reached_nodes):
        sources = []
        for src in dst2srcs[node]:
            if src in reached_nodes:
                continue
            reached_nodes.add(src)
            sources.append(solver(src))
            sources.extend(traversal_sources_depth_first(src, dst2srcs, solver, reached_nodes))
        return sources

    def traversal_sources_width_first(node, dst2srcs, solver):
        # reached_nodes = set()
        sources = []
        q = Queue()
        q.put(node)
        while not q.empty():
            node = q.get()
            for node in set(dst2srcs[node]):
                if node not in sources:  # reached_nodes
                    q.put(node)
                    # reached_nodes.add(node)
                    sources.append(solver(node))
        return sources

    if prior == 'depth':
        return traversal_sources_depth_first(node, dst2srcs, solver, set())
    elif prior == 'width':
        return traversal_sources_width_first(node, dst2srcs, solver)
    else:
        raise ValueError()


def traversal_source_edges(node, dst2edges, prior='width', solver=lambda edge: edge, next_node=lambda edge: edge[0]):
    # def traversal_sources_depth_first(node, dst2edges, solver, finish_nodes):
    #     source_edges = []
    #     for edge in dst2edges[node]:
    #         src, dst, edge_info = edge
    #         source_edges.append(solver(edge))
    #         source_edges.extend(traversal_sources_depth_first(src, dst2edges, solver, finish_nodes))
    #     finish_nodes.add(node)
    #     return source_edges

    def traversal_sources_width_first(node, dst2edges, solver):
        finished_node = set()
        source_edges = []
        q = Queue()
        q.put(node)
        while not q.empty():
            cur_node = q.get()
            for edge in dst2edges[cur_node]:
                source_edges.append(solver(edge))
                next_node_ = next_node(edge)
                if next_node_ not in finished_node:
                    q.put(next_node_)
            finished_node.add(cur_node)  # walked all edge to cur_node
        return source_edges

    if prior == 'depth':
        # return traversal_sources_depth_first(node, dst2edges, solver)
        raise NotImplementedError
    elif prior == 'width':
        return traversal_sources_width_first(node, dst2edges, solver)
    else:
        raise ValueError()


class BaseGraph(object):
    def __init__(self, nodes=None, edges=None):
        if nodes is not None:
            for node in nodes:
                self.add_node(node)
        if edges is not None:
            for edge in edges:
                self.add_edge(*edge)

    def add_node(self, node):
        raise NotImplementedError

    def add_edge(self, *args):
        raise NotImplementedError

    def get_nodes(self):
        raise NotImplementedError

    def get_src2dsts(self):
        raise NotImplementedError

    def get_dst2srcs(self):
        raise NotImplementedError

    def get_src2edges(self):
        raise NotImplementedError

    def get_dst2edges(self):
        raise NotImplementedError

    def topology_sort(self):
        return topology_sort(self.get_nodes(), self.get_src2dsts())

    def traversal_source_nodes(self, node, prior='depth', solver=lambda node: node):
        """
        find all reachable source nodes of given node
            node: give node.
            prior: 'depth' or 'width', traversal method.
            solver: callback function for each reached node.
        """
        return traversal_source_nodes(node, self.get_dst2srcs(), prior, solver)

    def traversal_destination_nodes(self, node, prior='depth', solver=lambda node: node):
        return traversal_source_nodes(node, self.get_src2dsts(), prior, solver)

    def traversal_source_edges(self, node, prior='width', solver=lambda node: node):
        return traversal_source_edges(node, self.get_dst2edges(), prior, solver)

    def traversal_destination_edges(self, node, prior='width', solver=lambda node: node):
        return traversal_source_edges(node, self.get_src2edges(), prior, solver, next_node=lambda edge: edge[1])


class SimpleGraph(BaseGraph):
    def __init__(self, nodes=None, edges=None):
        # self.dst2srcs = defaultdict(list)
        self.src2dsts = OrderedDict()
        self.nodes = set()
        super(SimpleGraph, self).__init__(nodes, edges)

    def add_edge(self, src, dst):
        if src not in self.src2dsts:
            self.src2dsts[src] = []
        self.src2dsts[src].append(dst)
        self.nodes |= {src, dst}

    def add_node(self, node):
        self.src2dsts[node] = []
        self.nodes.add(node)

    def add_srcs_dst(self, srcs, dst):
        # self.dst2srcs[dst].extend(srcs)
        for src in srcs:
            self.src2dsts[src].append(dst)
        self.nodes |= set(srcs + [dst])

    def add_src_dsts(self, src, dsts):
        self.src2dsts[src].extend(dsts)
        self.nodes |= set([src] + dsts)

    # get attribution
    def get_nodes(self):
        return self.nodes

    def get_src_nodes(self):
        return set(self.src2dsts.keys())

    def get_dst_nodes(self):
        dsts = []
        for src, _dsts in self.src2dsts.items():
            dsts.extend(_dsts)
        return set(dsts)

    def get_start_nodes(self):
        return self.nodes - self.get_dst_nodes()

    def get_end_nodes(self):
        return self.nodes - self.get_src_nodes()

    # get map
    def get_src2dsts(self):
        return self.src2dsts

    def get_dst2srcs(self):
        dst2srcs = defaultdict(list)
        for src, dsts in self.src2dsts.items():
            for dst in dsts:
                dst2srcs[dst].append(src)
        return dst2srcs


Edge = namedtuple("Edge", ("src", "dst", "info"))


class EdgeInfoGraph(BaseGraph):
    def __init__(self, nodes=None, edges=None):
        self.edges = OrderedDict()
        self.nodes = set()
        super(EdgeInfoGraph, self).__init__(nodes, edges)

    def add_node(self, node):
        self.edges[node] = OrderedDict()
        self.nodes.add(node)

    def add_edge(self, src, dst, edge_info, check=True, **kwargs):
        if check and self.have_edge(src, dst):
            import traceback
            for trace in traceback.format_stack(limit=3)[::-1]:
                print(trace.strip('\n'))
            dataset = kwargs['dataset']
            print(src, dst)
            print(src, dataset.get_tokens_by_loc(src), "src")
            print(dst, dataset.get_tokens_by_loc(dst), "dst")
            print(edge_info, "edge_info")
            print()
            assert False, "repeated add same edge."
        if src not in self.edges:
            self.edges[src] = OrderedDict()
        self.edges[src][dst] = edge_info
        self.nodes |= {src, dst}

    def get_src2dsts(self):
        src2dsts = defaultdict(list)
        for src, dsts_edge_info in self.edges.items():
            src2dsts[src] = list(dsts_edge_info.keys())
        return src2dsts

    def get_dst2srcs(self):
        dst2srcs = defaultdict(list)
        for src, dsts_edge_info in self.edges.items():
            for dst, edge_info in dsts_edge_info.items():
                dst2srcs[dst].append(src)
        return dst2srcs

    def get_src2edges(self):
        src2edges = defaultdict(list)
        for src, dsts_edge_info in self.edges.items():
            for dst, edge_info in dsts_edge_info.items():
                src2edges[src].append(Edge(src, dst, edge_info))
        return src2edges

    def get_dst2edges(self):
        dst2edges = defaultdict(list)
        for src, dsts_edge_info in self.edges.items():
            for dst, edge_info in dsts_edge_info.items():
                dst2edges[dst].append(Edge(src, dst, edge_info))
        return dst2edges

    def get_dsts_of_src(self, src):
        return set(self.edges[src].keys())

    # get attribution
    def get_nodes(self):
        return self.nodes

    def get_src_nodes(self):
        return set(self.edges.keys())

    def get_dst_nodes(self):
        dsts = []
        for src_loc, dsts_info in self.edges.items():
            for dst, edge_type in dsts_info.items():
                dsts.append(dst)
        return set(dsts)

    def get_start_nodes(self):
        return self.nodes - self.get_dst_nodes()

    def get_end_nodes(self):
        return self.nodes - self.get_src_nodes()

    def have_edge(self, src, dst):
        return (src in self.edges) and (dst in self.edges[src])


if __name__ == "__main__":
    G = SimpleGraph()
    """
    E
    D->B->A
       ^  ^
       |  |
       C->
    """
    G.add_node("E")
    G.add_srcs_dst(["B", "C"], "A")
    G.add_srcs_dst(["C", "D"], "B")
    print("topology sort 1", topology_sort(G.get_nodes(), G.get_src2dsts()))
    print("topology sort 2", G.topology_sort())
    print("source of A", G.traversal_source_nodes('A'))
    print("destination of D", G.traversal_destination_nodes('D'))

    G = EdgeInfoGraph()
    G.add_node("E")
    G.add_edge("B", "A", None)
    G.add_edge("C", "A", None)
    G.add_edge("C", "B", None)
    G.add_edge("D", "B", None)
    print(topology_sort(G.get_nodes(), G.get_src2dsts()))
