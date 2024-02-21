"""
a.建立全dataset的TRG；
    定义指向关系 =, 传参，引用。
        - =   在args或content里面
        - 传参 super.args指向 records_map[super.name].args
        - 引用 建立引用边的时候可能涉及到对DU的使用, 找出所有的D, 后面相同的的token都建立到它的edge (TODO：是否移除)
    list dag 内置函数（!strcat #）等复杂节点可以看成一个节点

    Relation Graph: 获取 relation set
        - build:
        - use  :
    Inherit Graph: 获取

b.	提取跟指令相关的TRG子图 relation_trg；
    TRG里src, dst都在指令的relation set的record中的edge构成的子图
c.	找到子图TRG中的所有起始节点(value)
    作过src节点，但没有做过dst节点就是起始节点
    DU，起始节点不能是D
        bits<5> a;  (DU: a(D))
        let Inst{5-1} = a; (DU: a(U))
        会建立上面的a(D)流向下面a(U)的边，上面的a也是只有流出没有流入，但不是我们想要的value
d.	提取流：在子图TRG中提取起始节点出发的所有流向构成的子图DataGraph，提取它的终止节点(name的候选节点)
    => (start_node, [(end_node,end_edge), ...])
    除了终止节点还有一种情况：
        int a = b;
        int c = a;
    TODO:
        b -> a -> c 的引用被建立， 但a也应该作为属性节点被记录，所以最终应该要记录不是终止节点，而是属性节点
        start_node -> [(attr_node, attr_src_node), (attr_node, attr_src_node)]
e.	终止节点有三种以及判定：
    1)	Attribute - 等号左值, 最后一条边为等号
    2)	Expr - (supertoken, 右值/实参), 最后一条边为引用
    3)  形参 – 死代码, 最后一条边为传参
    => (start_node, [(end_node, end_node_type)...])
f.  name_value_map 是一对一映射，attribute类型终止节点的个数：
    1) 留下所有end_node_type == ATTRIBUTE的节点
       => mid_res = (start_node, [(end_node, ATTRIBUTE), ...])
    2) assert len(mid_res[1]) == 1, 一般但单进多出不出现
       => mid_res = (start_node, end_node)
       => mid_res = (attr_name=token_of_end_node, [(start_node, end_node), ...])
    3) 多进单出 / 多进多出,通过优先级选出start_node
        a)	出现在子类中的start_node高于出现在父类中的start_node
        b)	出现在后面的父类的start_node高于前面的父类的start_node
       => (attr_name, (start_node, end_node)) => (name, value_loc)

g. 选择出需要调整的属性，消除依赖关系，收集 filter_name_value_map, filter_non_attr_map
    有些属性的类型定义再公用td里，提供的输入可能没有类型定义，不太好通过类型判断是否存在依赖，干脆直接通过属性名替换。
    而那些没有流向属性的输入，一定会先流向某个变量，这个变量往往作为形参拥有类型。
    1) filter_name_value_map：
        收集name_value_map通过优先级过滤掉的属性, (attr_name, (start_node, end_node)) => (name, [value_loc1, value_loc2])
    2) filter_non_attr_map：
        收集 所有end_node的end_node_type != ATTRIBUTE 的 start_node, 记录类型，
        (start_node, (next_node, type)) => (type, [(start_node, next_node), ...]) => (type, [value_loc1, value_loc2])

# g. 收集需要置为默认值的属性，需要删除的属性，必须给值的属性
#     # 直接置为默认值的属性：
#     {
#         "Itinerary":"NoItinerary",
#         "Pattern":[],
#
#         # 流向supertoken中的类型：?
#     }
"""



from parser_util_v2.dataset_constructor import Record, SuperToken, Location, Token, TokenPos
from collections import defaultdict
from enum import Enum
from parser_util_v2.project_utils import LOG
from queue import Queue
from third_utils.graph_utils import SimpleGraph, EdgeInfoGraph
from copy import deepcopy
import warnings
from collections import namedtuple
from parser_util_v2.dataset_constructor import Dataset
from parser_util_v2.get_relation_set import set_some_du_and_color, get_def_table
from tqdm import tqdm


def get_location(t):
    """
    get location of list of token.
    """

    def get_loc(t, idx):
        if isinstance(t, list):
            return get_loc(t[idx], idx)
        elif isinstance(t, Token):
            return t.loc
        else:
            raise TypeError(f'{type(t)}')

    if isinstance(t, Token):
        return t.loc
    loc_s, loc_e = get_loc(t, 0), get_loc(t, -1)
    return Location.create_location(loc_s.filename, loc_s.record_id, loc_s.token_id,
                                    length=loc_e.token_id - loc_s.token_id + 1)


def set_du(t, du, dataset):
    """
    set a list of Token's du to du
    DU:
        'D': defined
        'U': use
        'PD': partial defined, when define is a super token, each token.du set to 'PD'.
            such as Inst{3-4} of (Inst{3-4}=opcode;)
        'PU': partial used, when use is a super token, each token.du set to 'PU'.
            such as opstr of (!strconcat(opstr, t);)
        'T': Type of (let Type a = 1;) or (<Type a>)
        'IU': ignore use, def A:B<xxx> but lost define of B(an error input), here xxx set to 'IU'
    """
    # if du == 'D':
    #     assert isinstance(t, Token), f'excepted Token, but {type(t)} got'
    if isinstance(t, Token):
        if t.du is not None and t.du != 'T':  # 'T' is set for Type of (let Type a = 1;) in build_graph
            if t.du != du:  # f'{t}({t.du}) {dataset.get_record(t.loc)}'
                raise ValueError()
        else:
            t.du = du
    elif isinstance(t, list):
        if len(t) > 1 and du == 'D':
            du = 'PD'
        elif len(t) > 1 and du == 'U':
            du = 'PU'
        for st in t:
            set_du(st, du, dataset)
    else:
        raise TypeError()


def is_use(t):
    return t.du == 'U' or t.du == 'IU' or t.du == 'PU'


def is_full_def(t):
    return t.du == 'D'


def is_def(t):
    return t.du == 'D' or t.du == 'PD'


class EdgeType(Enum):
    # EQUAL = 0  # A = B
    ARG_EQUAL = 0
    STMT_EQUAL = 1
    INHERIT = 2  # A : B
    PARAM = 3  # B<T x>; B<'add'>  ('add'=>x)
    IDENTITY = 4  # same token string in different loc


class TokenReferenceGraph(EdgeInfoGraph):
    def __init__(self, dataset):
        super(TokenReferenceGraph, self).__init__()
        self.dataset = dataset
        self.edge_group = None

    def add_edge_with_token(self, src, dst, edge_type, check=True):
        self.edge_group = None  # if graph update, edge_group need update if used.
        st = (get_location(src), get_location(dst))
        # assert isinstance(src, (Token, SuperToken)) or isinstance(dst, (Token, SuperToken)), f"{src} {dst}"
        if check and self.have_edge(st[0], st[1]):
            import traceback
            print(traceback.format_stack(limit=3))
            print(st, src, dst, edge_type)

        super(TokenReferenceGraph, self).add_edge(st[0], st[1], edge_type, check, dataset=self.dataset)

        if edge_type != EdgeType.IDENTITY:
            set_du(dst, 'D', self.dataset)
            set_du(src, 'U', self.dataset)

    def set_dus(self, tokens_list, du_list):
        for tokens, du in zip(tokens_list, du_list):
            set_du(tokens, du, self.dataset)

    # a.
    def build_on_dataset(self):
        """
        build graph and set DU
        1. loc->[tuple(loc1_1, loc1_2), tuple(loc2_1, loc2_2)....]
        Assign:
            A=B;  (B=>A)
            A:B;  (B=>A)
            A: B<'add'>;  B<string name>; ('add' => name)
        等值传递(DU):
            A = B; C = A;  (A=>A)
            A<string name>: B<name>; (name=>name)
            A<string name>{ let opstr=name;} (name=>name)
        除了上面会产生DU之外，还有能产生DU的地方
            def R;      // in function
            string A;   // in args or in content
        name
        """

        def build_assign_edge(trg, dataset):
            # deal 3 kind of assignment
            not_found_name = []
            for filename, records in dataset:
                for record in records:
                    trg.set_dus([record.data['name']], ['D'])  # class R;
                    # # A:B
                    # for super_name, _ in record.data['supers']:
                    #     trg.add_edge_with_token(super_name, record.data['name'], EdgeType.INHERIT)
                    # A=B
                    for var_type, var_name, value in record.data['args']:
                        if value is not None:
                            trg.add_edge_with_token(value, var_name, EdgeType.ARG_EQUAL)
                        # string A;
                        trg.set_dus([var_name, var_type], ['D', 'T'])
                    for var_type, var_name, value in record.data['content']:
                        if value is not None:
                            trg.add_edge_with_token(value, var_name, EdgeType.STMT_EQUAL)
                        # string A;
                        trg.set_dus([var_name, var_type], ['D', 'T'])
                    # B<>; A<>:B<>;
                    for super_name, super_args in record.data['supers']:
                        if super_name not in self.dataset.records_map:  # that should be a bug
                            not_found_name.append(super_name)
                            trg.set_dus(super_args, ['IU'] * len(super_args))
                            continue
                        super_record = self.dataset.records_map[super_name]
                        for use_arg, def_arg in zip(super_args, super_record.data['args']):
                            _, def_var, _ = def_arg
                            trg.add_edge_with_token(use_arg, def_var, EdgeType.PARAM)
                    # if record[1] == 'MipsInst':
                    #     record.print(with_du=True)
            if len(not_found_name) > 0: LOG.warn(set(not_found_name), 'as superclass not found')

        def build_identifier_edge(trg, dataset):
            cross_record_defined_token = {}  # {rname: rname for rname in
            #   list(self.dataset.records_map.keys())}  # define and defined
            # A=B; C=A; (A=>A)
            for filename, records in dataset:
                for record in records:
                    # collect all defined token
                    defined_token = {}
                    for token in record:
                        if is_full_def(token):
                            defined_token[token] = token
                    defined_token.update(cross_record_defined_token)
                    # link the define and use of same token string
                    for token in record:
                        if is_use(token) and token in defined_token:
                            # for Identifier, data from define to use
                            trg.add_edge_with_token(defined_token[token], token, EdgeType.IDENTITY)

        set_some_du_and_color(self.dataset)
        # global_def_table = get_def_table(self.dataset)  # for relation set find
        build_assign_edge(self, self.dataset)
        build_identifier_edge(self, self.dataset)

    # function to get sub token reference graph with given records_name
    def group_edges(self):
        self.edge_group = defaultdict(list)  # src_record_id: list(edge)
        src2edges = self.get_src2edges()
        for src_loc, edges in src2edges.items():
            src_record_fig = self.dataset.get_record_fig(src_loc)
            for edge in edges:
                dst_record_fig = self.dataset.get_record_fig(edge[1])
                self.edge_group[(src_record_fig, dst_record_fig)].append(edge)

    # b.
    def sub_graph_of_records(self, records_name):
        if self.edge_group is None:
            self.group_edges()
            # raise ValueError("please call group_edges() firstly before call sub_graph_of_records(...)")
        sub_trg = TokenReferenceGraph(self.dataset)
        for name in records_name:
            record_fig = self.dataset.records_fig[name]
            for dst_name in records_name:
                dst_record_fig = self.dataset.records_fig[dst_name]
                for edge in self.edge_group[(record_fig, dst_record_fig)]:
                    # edge 是 (src_loc, dst_loc, edge_info) 不是（src, dst, edge_info)，使用sub_trg.add_edge不行
                    super(TokenReferenceGraph, sub_trg).add_edge(*edge, check=True, dataset=self.dataset)
        return sub_trg

    # c. 找到子图TRG中的所有起始节点(value)
    def get_start_use_locations(self):
        def is_use_filter(start_locs):
            return [start_loc for start_loc in start_locs if is_use(self.dataset.get_tokens_by_loc(start_loc)[0])]

        # def not_inherit_filter(start_locs, src_to_edges):
        #     # dst_to_start_edges = defaultdict(list)
        #     new_start_locs = []
        #     for start_loc in start_locs:
        #         edges = src_to_edges[start_loc]
        #         # 开始的输入参数只能传给一个位置
        #         assert len(edges) == 1, [self.edge_to_str(e) for e in edges]
        #         _, dst_loc, edge_type = edges[0]
        #         if edge_type == EdgeType.INHERIT:  # 防止一些公共类由于没有给出定义文件而被误认为是输入，比如InstrItinClass
        #             continue
        #         new_start_locs.append(start_loc)
        #         # dst_to_start_edges[dst_loc].append(edges[0])
        #     return new_start_locs

        # def filename_filter(start_locs):
        #     locs = []
        #     for loc in start_locs:
        #         keep = True
        #         for filename in ["target\Target.td"]:
        #             if loc.filename.endswith(filename):
        #                 keep = False
        #         if keep:
        #             locs.append(loc)
        #     return locs

        start_locs = self.get_start_nodes()  # 只作为src，不作为dst，即是开头

        start_locs = is_use_filter(start_locs)
        # start_locs = not_inherit_filter(start_locs, self.get_src2edges())
        # start_locs = filename_filter(start_locs)
        return start_locs

    # d. 提取流：在子图TRG中提取起始节点出发的所有流向构成的子图DataGraph，提取它的终止节点(name的候选节点)
    def sub_graph_of_start_node(self, start_node, **kwargs):
        src2edges = self.get_src2edges()
        sub_trg = DataGraph(self.dataset, start_node)
        walked_nodes = set()
        node_stack = [start_node]
        while len(node_stack) > 0:
            node = node_stack.pop()
            if len(src2edges[node]) == 0:
                sub_trg.end_nodes.append(node)
            for src, dst, edge_info in src2edges[node]:
                sub_trg.add_edge(src, dst, edge_info, **kwargs)
                if dst not in walked_nodes:
                    node_stack.append(dst)
            walked_nodes.add(node)
        return sub_trg

    # functions for debug
    def edge_to_str(self, edge, with_loc=False):
        src_loc, dst_loc, edge_type = edge
        link = '--' if edge_type == EdgeType.IDENTITY else '->'
        src = self.dataset.get_tokens_by_loc(src_loc)
        dst = self.dataset.get_tokens_by_loc(dst_loc)
        # print(self.dataset.get_record(src_loc))
        # print(src_loc)
        dst_du, src_du = dst[0].du, src[0].du
        return f'{"".join(src)}({src_du}) {link} {"".join(dst)}({dst_du})' + \
               (f'\n{src_loc} {link} {dst_loc}' if with_loc else '')

    def node_to_str(self, node, with_loc=False):
        node_tokens = self.dataset.get_tokens_by_loc(node)
        return f'{"".join(node_tokens)}({node_tokens[0].du})' + (f'\n{node}' if with_loc else '')

    def print(self, concert_rs=None, with_loc=False):
        for src_loc, dsts_info in self.edges.items():
            for dst_loc, edge_type in dsts_info.items():
                if concert_rs is None or self.dataset.get_record(src_loc).data['name'] in concert_rs or \
                        self.dataset.get_record(dst_loc).data["name"] in concert_rs:
                    print(self.edge_to_str((src_loc, dst_loc, edge_type), with_loc))


class DataGraph(TokenReferenceGraph):
    def __init__(self, dataset, start_node):
        super(DataGraph, self).__init__(dataset)
        self.start_node = start_node
        self.end_nodes = []

    def get_end_edges2(self):
        end_edges = []
        end_nodes = self.get_end_nodes()
        for src, dst_info in self.edges.items():
            for dst, edge_info in dst_info.items():
                # print("edge info", src, dst, edge_info)
                if dst in end_nodes:
                    # print("catch dst", dst)
                    # Attribute =
                    end_edges.append((src, dst, edge_info))
        return end_edges

    def get_attr_end_edges(self):
        attr_edges = []

        def solver(edge):
            if edge[2] == EdgeType.STMT_EQUAL:
                attr_edges.append(edge)

        self.traversal_destination_edges(self.start_node, solver=solver)
        return list(set(attr_edges) | set(self.get_end_edges()))

    def get_end_edges(self):
        end_edges = []
        end_nodes = self.get_end_nodes()
        dst2edges = self.get_dst2edges()  # => edge = (src, dst, info) => dst2edges = {dst: [edge, ...]}
        for end_node in end_nodes:
            edges = dst2edges[end_node]
            assert len(edges) == 1, f"{edges}, an end node have multiple edge to it from same start node."
            end_edges.append(edges[0])
        return end_edges

    def get_end_node_type(self, end_edge):
        src, dst, edge_info = end_edge
        if edge_info == EdgeType.ARG_EQUAL or edge_info == EdgeType.STMT_EQUAL:
            return EndNodeType.ATTRIBUTE
        # 2 Expr
        elif edge_info == EdgeType.IDENTITY:
            return EndNodeType.EXPR
        # 3 PARAM
        elif edge_info == EdgeType.PARAM:
            return EndNodeType.FORMAT_ARG
        else:
            raise TypeError(end_edge)

    # old API
    def get_last_def_loc(self):
        dst2srcs = self.get_dst2srcs()
        last_def_loc = set()
        for node in self.end_nodes:
            if is_def(self.dataset.get_tokens_by_loc(node)[0]):
                last_def_loc.add(node)
            else:
                srcs = dst2srcs[node]
                assert len(srcs) == 1 and is_def(self.dataset.get_tokens_by_loc(srcs[0])[0]), \
                    "if end node is use, it must from def node"
                last_def_loc.add(srcs[0])
        return last_def_loc

    def get_last_def_loc2(self):
        last_def_nodes = {}
        src2edges = self.get_src2edges()
        end_nodes = set()
        node_stack = [self.start_node]
        while len(node_stack) > 0:
            node = node_stack.pop()
            for src, dst, edge_info in src2edges[node]:
                if is_def(self.dataset.get_tokens_by_loc(dst)[0]):
                    last_def_nodes[dst] = dst
                else:
                    last_def_nodes[dst] = last_def_nodes[src]
                node_stack.append(dst)
            if len(src2edges[node]) == 0:
                end_nodes.add(node)
        return set([last_def_nodes[node] for node in end_nodes])

    @staticmethod
    def get_last_def_loc0(node, src2edges, global_trg_node_stat, dataset, src_node=None):
        """
        如果start_loc存在大量的重合，那么这个函数通过动态规划存储每个节点的结果于global_trg_node_stat，将发挥极大优势。
        last_defs[src] = set([last_defs[dst] for dst in src2dsts[src])
        """
        last_defs = set()
        edges = src2edges[node]
        for node, dst, edge_info in edges:
            stat = global_trg_node_stat[dst]
            if stat.find_last_def:
                last_defs |= stat.last_defs
            else:
                last_defs |= DataGraph.get_last_def_loc0(dst, src2edges, global_trg_node_stat, dataset, src_node=node)

        if len(edges) == 0:  # node is end_node
            if is_def(dataset.get_tokens_by_loc(node)[0]):
                last_defs.add(node)
            else:
                assert is_def(dataset.get_tokens_by_loc(src_node)[0])
                last_defs.add(src_node)

        global_trg_node_stat[node] = NodeStat(True, last_defs)
        return last_defs


# class RelationGraph(SimpleGraph):
#     def __init__(self, dataset):
#         super(RelationGraph, self).__init__()
#         self.dataset = dataset
#
#     # relation set
#     def build_relation_graph(self, trg):
#         def is_from_local_src(token, dst_to_edges):
#             edges = dst_to_edges[token.loc]
#             assert len(edges) == 1, f"{token} should be 'U' or 'IU', which only have one src. {edges}"
#             # A=B; C=A; , A(in C=A) is local use
#             for src_loc, _, _ in edges:
#                 if (src_loc.filename, src_loc.record_id) != (
#                 token.loc.filename, token.loc.record_id):  # not in same use
#                     return False
#             return True
#
#         def is_global_use(token, dst_to_edges):
#             return is_use(token) and (not is_from_local_src(token, dst_to_edges))
#
#         dst2edges = trg.get_dst2edges()
#         for record_name, records in self.dataset.records_map.items():
#             for t in records:
#                 if t in self.dataset.records_map and is_global_use(t, dst2edges):
#                     self.add_edge(t, record_name)
#
#     def get_relation_set(self, record_names, get_loc=True):
#         record_names = set(record_names)
#         relation_sets = deepcopy(record_names)
#         for inst_name in record_names:
#             relation_sets |= self.traversal_source_nodes(inst_name)
#
#         if get_loc:
#             relation_sets = {self.dataset.records_map[name].data['name'].loc for name in relation_sets}
#         return relation_sets


class RelationGraph(EdgeInfoGraph):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        super(RelationGraph, self).__init__()

    def build_on_dataset(self):
        for rname, record in self.dataset.records_map.items():
            for token in record:
                if token == rname:
                    continue
                if token in self.dataset.records_map and token not in ['B', 'LI']:  # fix bits name same with rname
                    src, dst = rname, str(token)
                    if not self.have_edge(src, dst):
                        """
                        class B<InstType inst_type = InstTypeA>{
                                InstType Inst_type = inst_type;
                        }
                        """
                        self.add_edge(src, dst, None, check=True, dataset=self.dataset)

    def get_relation_record_names(self, record_name):
        return [record_name] + self.traversal_destination_nodes(record_name, prior='width')


class InheritGraph(EdgeInfoGraph):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        super(InheritGraph, self).__init__()

    def build_on_dataset(self):
        for rname, record in self.dataset.records_map.items():
            assert record[0] != 'let', "let constraint make priority compare difficult"
            for super_name, super_args in record.data['supers'][::-1]:
                self.add_edge(rname, str(super_name), None, check=True, dataset=self.dataset)

    def get_accent_record_names(self, record_name):
        return self.traversal_destination_nodes(record_name)

    def traversal_node_with_priority(self, rname):
        """
        def A:B,C;
        prio: A>B>C
        """
        node_list = [rname]
        for super, _ in self.dataset.records_map[rname].data["supers"][::-1]:
            if super in self.dataset.records_map.keys():
                node_list.extend(self.traversal_node_with_priority(super))
            else:
                continue
        return node_list

    def traversal_node_with_priority2(self, rname):
        """
        def A:B,C;
        prio: A>B>C
        """
        return [rname] + self.traversal_destination_nodes(rname, prior='depth')

    def get_node_with_priority(self, rname):
        node_list = self.traversal_node_with_priority(rname)
        node2priority = {}
        for i, node in enumerate(node_list[::-1]):
            node2priority[node] = i

        return node2priority

    def priority_gt(self, a, b, node2priority, equal_class_warning=False):
        if isinstance(a, Location):
            loc_a = a
            a = self.dataset.get_record(a).data["name"]
        if isinstance(b, Location):
            loc_b = b
            b = self.dataset.get_record(b).data["name"]
        assert isinstance(a, str) and isinstance(b, str), (a, b)

        if node2priority[a] == node2priority[b]:
            if equal_class_warning:
                print(__file__, "node priority warning")
                print(loc_a, self.dataset.get_tokens_by_loc(loc_a), a)
                print(loc_b, self.dataset.get_tokens_by_loc(loc_b), b)
            return loc_a.token_id > loc_b.token_id  # 没有let在头
            # raise ValueError

        return node2priority[a] > node2priority[b]

    def priority_compare(self, start_a, end_a, start_b, end_b, node2priority):
        if end_a != end_b:
            return self.priority_gt(end_a, end_b, node2priority)
        else:  # end 相同，比较start
            # 都相同就是一个节点，选谁都行，这个assert可删除
            assert start_a != start_b
            return self.priority_gt(start_a, start_b, node2priority, equal_class_warning=True)


NodeStat = namedtuple("NodeStat", ["find_last_def", "last_defs"])


# class GraphHandler(object):
#     def __init__(self, dataset):
#         self.dataset = dataset
#
#         self.token_ref_graph = TokenReferenceGraph(dataset)
#         self.relation_graph = RelationGraph(dataset)
#
#         self.global_trg_node_stat = {}
#
#         self.accents = {}
#         self.children = {}
#
#     def build_graph(self):
#         self.token_ref_graph.build_on_dataset()
#         self.relation_graph.build_relation_graph(self.token_ref_graph)
#
#         for node in self.token_ref_graph.nodes:
#             self.global_trg_node_stat[node] = NodeStat(False, set())
#
#         self.build_inherit()
#
#     def build_inherit(self):
#         supers = {rn: set() for rn in self.dataset.records_fig.values()}
#         children = {rn: set() for rn in self.dataset.records_fig.values()}
#         for record_name in self.dataset.records_map:
#             r = self.dataset.records_map[record_name]
#             r_fig = self.dataset.records_fig[record_name]
#             for super_name, super_args in r.data['supers']:
#                 if super_name in self.dataset.records_fig:
#                     supers[r_fig].add(self.dataset.records_fig[super_name])
#                     children[self.dataset.records_fig[super_name]].add(r_fig)
#
#         def get_accents(supers, r_figs, accents):
#             """
#             return set(r_figs) | set(accents of any one in r_figs)
#             """
#             the_supers = r_figs.copy()
#             for r_fig in r_figs:
#                 if r_fig not in accents:
#                     accents[r_fig] = get_accents(supers, supers[r_fig], accents)
#                 the_supers |= accents[r_fig]
#             return the_supers
#
#         self.children = children
#         full_records = get_accents(supers, set(supers.keys()), self.accents)
#         assert len(full_records) == len(self.dataset.records_map)
#
#     def get_relation_set(self, record_names, get_loc=True):
#         return self.relation_graph.get_relation_set(record_names, get_loc)
#
#     def get_name_value_loc_of_records(self, name_value_loc, record_names, inst_name):
#         def delete_default_value_if_have_param_value(name_loc_to_value_loc):
#             # get edge_type of each start_loc
#             src2edges = relation_trg.get_src2edges()
#             start_loc_edge_type = {}
#             for start_loc in start_locs:
#                 edges = src2edges[start_loc]
#                 assert len(edges) == 1
#                 start_loc_edge_type[start_loc] = edges[0][-1]
#
#             # if name_loc have value pass from param, then delete all default value(assign by equal)
#             new_name_loc_to_value_loc = {}
#             for name_loc, value_locs in name_loc_to_value_loc.items():
#                 have_param_src = False
#                 for loc in value_locs:
#                     if start_loc_edge_type[loc] == EdgeType.PARAM:
#                         have_param_src = True
#                 if have_param_src:
#                     value_locs = [loc for loc in value_locs if start_loc_edge_type[loc] != EdgeType.EQUAL]
#                     for loc in value_locs:
#                         assert start_loc_edge_type[loc] == EdgeType.PARAM
#                 new_name_loc_to_value_loc[name_loc] = value_locs
#             return new_name_loc_to_value_loc
#
#         if isinstance(list(record_names)[0], Record):
#             record_names = [r.data["name"] for r in record_names]
#
#         relation_trg = self.token_ref_graph.sub_graph_of_records(record_names)
#         relation_trg_records_name = list(set([self.dataset.get_record(node).data['name'] for node in relation_trg.nodes]))
#         assert tuple(sorted(record_names)) == tuple(sorted(relation_trg_records_name))
#         start_locs = relation_trg.get_start_use_locations()
#
#         # start_loc => names_loc
#         start_loc2names_loc = {}
#         for start_loc in sorted(start_locs):
#             # print(start_loc, self.dataset.get_tokens_by_loc(start_loc))
#             # last_def_loc = DataGraph.get_last_def_loc0(start_loc, relation_trg.get_src2edges(),
#             #                                            self.global_trg_node_stat, self.dataset)
#             data_graph = relation_trg.sub_graph_of_start_node(start_loc, dataset=self.dataset)
#             last_def_loc = data_graph.get_last_def_loc()
#             start_loc2names_loc[start_loc] = last_def_loc
#
#         # names_loc => start_loc
#         name_loc_to_value_loc = defaultdict(list)
#         for start_loc, last_def_locs in start_loc2names_loc.items():
#             names = tuple(sorted(last_def_locs))
#             name_loc_to_value_loc[names].append(start_loc)
#
#         name_loc_to_value_loc = delete_default_value_if_have_param_value(name_loc_to_value_loc)
#
#         def name1_record_inherit_position(loc1):
#             def find_in_tree(name, want_name, idxs):
#                 if name not in self.dataset.records_map:
#                     return False
#                 r = self.dataset.records_map[name]
#                 for i, (super_name, _) in enumerate(r.data['supers']):
#                     if super_name == want_name or find_in_tree(super_name, want_name, idxs):
#                         idxs.append(i)
#                         return True
#                 return False
#
#             record_name1 = self.dataset.get_record(loc1).data['name']
#             r = self.dataset.records_map[inst_name]
#             # print("record name", record_name1, inst_name, r.data['supers'])
#             idxs = []
#             for i, (super_name, _) in enumerate(r.data['supers']):
#                 if super_name == record_name1 or find_in_tree(super_name, record_name1, idxs):
#                     idxs.append(i)
#                     return idxs[::-1]
#             assert False, f"{record_name1} not in super class of {inst_name}"
#
#         def pos1_gt_pos2(pos1, pos2):
#             for p1, p2, in zip(pos1, pos2):
#                 if p1 > p2:
#                     return True
#                 elif p1 < p2:
#                     return False
#             assert False, f"{pos1} vs {pos2}"
#
#         def name1_higher_prior_than_name2(name1, name2, loc1, loc2):
#             """
#             name same different value location
#             1. r1: r2 one in super class, one in child class record. => child cover super
#             2. in same class Record => later one cover before one
#             """
#             assert name1 == name2
#             name1_loc, name2_loc = name1.loc, name2.loc
#             if name1_loc == name2_loc:
#                 assert False
#             else:
#                 r1_fig = self.dataset.get_record_fig(name1_loc)
#                 r2_fig = self.dataset.get_record_fig(name2_loc)
#                 if r2_fig in self.accents[r1_fig]:   # r1 : r2
#                     return True
#                 if r1_fig in self.accents[r2_fig]:   # r2 : r1
#                     return False
#                 if r1_fig == r2_fig:                # same class record
#                     assert name1_loc.line_id != name2_loc.line_id
#                     return name1_loc.line_id > name2_loc.line_id
#
#                 # 重复定义来自不同的继承链，使用后面来自基础父类的属性覆盖前面的基础父类
#                 pos1 = name1_record_inherit_position(name1_loc)  # 所在的record是inst_name的第几个父类的祖先
#                 pos2 = name1_record_inherit_position(name2_loc)
#                 if pos1_gt_pos2(pos1, pos2):
#                     return True
#                 elif pos1_gt_pos2(pos2, pos1):
#                     return False
#                 print(r1_fig, r2_fig, name1, name2, pos1, pos2)
#                 print("name1", self.dataset.get_record(name1_loc))
#                 print("name2", self.dataset.get_record(name2_loc))
#                 print(f"{name1_loc}, {name1}\n{loc1[0]}\n{name2_loc}, {name2}\n{loc2[0]}")
#                 assert False
#
#         def set_key_value(nv_loc, name, value_locs, full_nv_loc, name2name):
#             full_nv_loc[name].append(value_locs)
#             # if name in nv_loc:
#             #     name1, loc1 = name2name[name], nv_loc[name]
#             #     name2, loc2 = name, value_locs
#             #     if name1_higher_prior_than_name2(name1, name2, loc1, loc2):
#             #         return
#             nv_loc[name] = value_locs
#             name2name[name] = name
#
#         # name => start_loc
#         nv_loc = {}
#         full_nv_loc = defaultdict(list)
#         name2name = {}
#         for name_locs, value_locs in name_loc_to_value_loc.items():
#             names = []
#             for name_loc in name_locs:
#                 name = self.dataset.get_tokens_by_loc(name_loc)
#                 if len(name) > 1:
#                     assert name[0] in ['Inst', 'TSFlags', 'HWEncoding', 's', 's13', 's8', 'u5', 'fieldB', 'fieldU', 'off', 's11'], name
#                 names.append(name[0])
#             name = "+".join(names) if len(names) > 1 else names[0]
#
#             if name in ['Inst', 'TSFlags', 'HWEncoding', 's', 's13', 's8', 'u5', 'fieldB','fieldU', 'off', 's11']:  # 0x201=>funct; Inst{x-y} = funct;  => Inst_funct: 0x201
#                 assert len(value_locs) == 1
#                 next_name_loc = list(relation_trg.get_dsts_of_src(value_locs[0]))
#                 assert len(next_name_loc) == 1, next_name_loc
#                 next_name = self.dataset.get_tokens_by_loc(next_name_loc[0])
#                 if next_name[0] == name:
#                     key = "".join(next_name).replace('{', "").replace("}", "").replace("-", "_") + '_*v*'
#                     nv_loc[key] = value_locs
#                     # set_key_value(nv_loc, key, value_locs, full_nv_loc, name2name)
#                 else:
#                     assert len(next_name) == 1, next_name
#                     key = f"{name}_{next_name[0]}_*v*"
#                     assert key not in nv_loc
#                     nv_loc[key] = value_locs
#                     # set_key_value(nv_loc, key, value_locs, full_nv_loc, name2name)
#                 if self.dataset.get_record(value_locs[0]).data['name'] == 'ADD':
#                     print(inst_name, key, self.dataset.get_tokens_by_loc(value_locs[0]))
#             else:
#                 set_key_value(nv_loc, name, value_locs, full_nv_loc, name2name)
#
#         return nv_loc, {"start_loc2_names_loc": start_loc2names_loc, "name2start_loc": nv_loc,
#                         "dataset": self.dataset, "full_nv_loc": full_nv_loc}
#
#
#     def get_input_name_value_loc(self):
#         return {}
#
#     def print_name_value_loc(self, name_value_loc):
#         print(len(name_value_loc))
#         for name_loc, value_locs in name_value_loc.items():
#             print(name_loc, '\t', self.dataset.get_tokens_by_loc(name_loc), '<-',
#                   [self.dataset.get_tokens_by_loc(value_loc) for value_loc in value_locs])
#
#     def check(self):
#         self.check_location()
#         self.check_du()
#         self.check_token_type()
#         self.check_record_name()
#
#     def check_du(self):
#         keywords = ['def', 'class', 'let', 'foreach', 'multiclass', 'defm', 'bits', 'string', 'bit', 'dag', 'list']
#         for filename, records in self.dataset:
#             for r in records:
#                 for t in r:
#                     if t.isidentifier() and (t.du is None) and t not in keywords:
#                         print(t.loc, t, r)
#                         assert False
#
#     def check_location(self):
#         L = {}
#         for filename, records in self.dataset:
#             for r in records:
#                 for t in r:
#                     assert t.loc not in L, f"{t.loc} {t} {r}"
#                     L[t.loc] = r
#
#     def check_token_type(self):
#         for filename, records in self.dataset:
#             for r in records:
#                 for t in r:
#                     assert isinstance(t, Token), f"{t} {type(t)}"
#
#     def check_record_name(self):
#         record_map = set()
#         for filename, records in self.dataset:
#             for r in records:
#                 r_name = r.data["name"]
#                 if r_name not in record_map:
#                     record_map.add(r_name)
#                 else:
#                     raise ValueError(f"{r_name}")
#                 assert r_name.loc == r.data["name"].loc
#
#     # def check_type(self):
#
#     # old version function TODO(delete)
#     # the formation of graph can be data_flow, src_to_edges, dst_to_edges
#     def get_data_flow(self):
#         # src_loc = > [(dst_loc,), ...]
#         data_flow, edge_type_dict = defaultdict(list), defaultdict(list)
#         for src_loc, dsts_info in self.token_ref_graph.edges.items():
#             for dst_loc, edge_type in dsts_info.items():
#                 data_flow[src_loc].append((dst_loc,))
#                 edge_type_dict[src_loc].append(edge_type)
#         return data_flow, edge_type_dict
#
#     def get_input_name_value_loc(self):
#         start_locs = self.token_ref_graph.get_start_use_locations()
#         # r = self.records_map['MipsInst']
#         # for loc in start_locs:
#         #     if self.dataset.get_record(loc) == r:
#         #         s = self.dataset.get_tokens_by_loc(loc)
#         #         print(loc, s)
#         # print(len(start_locs))
#
#         # dst_loc: [start_edge, ...], start_edge means a edge whose src_loc is in start_locs
#         src_to_edges = self.token_ref_graph.get_src2edges()
#         dst_to_start_edges = defaultdict(list)
#         for start_loc in start_locs:
#             edges = src_to_edges[start_loc]
#             assert len(edges) == 1, [self.token_ref_graph.edge_to_str(e) for e in edges]
#             _, dst_loc, edge_type = edges[0]
#             # assert edge_type != EdgeType.INHERIT, self.edge_to_str(edges[0])
#             if edge_type == EdgeType.INHERIT:
#                 continue
#             dst_to_start_edges[dst_loc].append(edges[0])
#
#         name_loc_to_value_loc = {}
#         for dst_loc, edges in dst_to_start_edges.items():
#             # have_param_src means need to delete all default values.
#             have_param_src = False
#             for edge in edges:
#                 if edge[-1] == EdgeType.PARAM:
#                     have_param_src = True
#             if have_param_src:
#                 edges = [edge for edge in edges if edge[-1] != EdgeType.EQUAL]
#                 for edge in edges:
#                     assert edge[-1] == EdgeType.PARAM
#             name_loc_to_value_loc[dst_loc] = [edge[0] for edge in edges]
#         return name_loc_to_value_loc
#
#     def get_name_value_loc_of_records2(self, name_value_loc, relation_records):
#         # self.print_name_value_loc(name_value_loc)
#         relation_record_locs = [(r[0].loc.filename, r[0].loc.record_id) for r in relation_records]
#         # for rloc, rr in zip(relation_record_locs, relation_records):
#         #     print(rloc, rr)
#         # print()
#
#         relation_nv_loc = {}
#         # relation_nv_loc2loc = {}
#         for name_loc, values_loc in name_value_loc.items():
#             if (name_loc.filename, name_loc.record_id) not in relation_record_locs:
#                 # print(name_loc, self.dataset.get_tokens_by_loc(name_loc), 2)
#                 continue
#             values_loc = [value_loc for value_loc in values_loc if
#                           (value_loc.filename, value_loc.record_id) in relation_record_locs]
#             if len(values_loc) == 0:
#                 # print(name_loc, self.dataset.get_tokens_by_loc(name_loc), 3)
#                 continue
#             name = self.dataset.get_tokens_by_loc(name_loc)
#             if len(name) > 1:
#                 if name[0] not in ['Inst', 'TSFlags', 'HWEncoding', 's', 's13', 's8', 'fieldB', 'fieldU', 'off', 's11']:
#                     LOG.warn(name)
#                 continue
#             name = name[0]
#             # print(name_loc, self.dataset.get_tokens_by_loc(name_loc), 4)
#             if name in relation_nv_loc:
#                 # print(relation_nv_loc[name][0])
#                 # print(values_loc[0])
#                 LOG.debug(name, "has already in dict")
#                 continue
#             # print(name_loc, self.dataset.get_tokens_by_loc(name_loc), name, 5)
#             relation_nv_loc[name] = values_loc
#             # relation_nv_loc2loc[name_loc] = values_loc
#         return relation_nv_loc# , relation_nv_loc2loc
#


def tmp_get_inst_names(relation_g, inst_names=None):
    """
        for current version: inst record will not be used by other records.
        for some inst(like used by Pat), it should be removed.
    """
    tmp_inst_names = relation_g.get_start_nodes()
    # if inst_names is not None:
    #     assert len(set(tmp_inst_names) - set(inst_names)) == 0
    #     assert len(set(inst_names) - set(tmp_inst_names)) == 0
    if inst_names is not None:
        for inst_name in inst_names:
            assert inst_name in tmp_inst_names
        tmp_inst_names = inst_names
    return tmp_inst_names


def debug_print(*args, **kwargs):
    # print(*args, **kwargs)
    pass


class EndNodeType(Enum):
    ATTRIBUTE = 0
    EXPR = 1
    FORMAT_ARG = 2


def get_type_loc_by_var_loc(loc, dataset):
    """
    loc 只可能在形参，contents两处
    """
    r = dataset.get_record(loc)
    for arg in r.data['args']:
        type_list, name, _ = arg
        assert len(name) == 1, name
        if loc == name[0].loc:
            return get_location(type_list)

    for content in r.data['content']:
        type_list, name, _ = content
        assert len(name) == 1, name
        if loc == name[0].loc:
            return get_location(type_list)


def tmp_remove_bits_edges(attr_nodes, dataset: Dataset):
    new_attr_modes = []
    for attr_node in attr_nodes:
        tokens = dataset.get_tokens_by_loc(attr_node)
        if tokens[0] not in ['Inst', 'TSFlags']:
            new_attr_modes.append(attr_node)
    return new_attr_modes


class GraphHandler(object):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.trg = TokenReferenceGraph(dataset)
        self.relation_g = RelationGraph(dataset)
        self.inherit_g = InheritGraph(dataset)

    # a.建立全dataset的TRG
    def build_graph(self):
        self.trg.build_on_dataset()
        self.relation_g.build_on_dataset()
        self.inherit_g.build_on_dataset()

    def get_relation_sets(self, inst_names):
        inst_names = tmp_get_inst_names(self.relation_g, inst_names)
        relation_record_names = {}
        for inst_name in inst_names:
            relation_record_names[inst_name] = self.relation_g.get_relation_record_names(inst_name)[::-1]
        return relation_record_names

    def run(self, inst_names):
        inst_names = tmp_get_inst_names(self.relation_g, inst_names)
        name_value_map = {}
        filter_name_value_map = {}
        filter_non_attr_map = {}
        for inst_name in sorted(inst_names):
            # print(inst_name)
            # b. 提取跟指令相关的TRG子图 relation_trg
            relation_record_names = self.relation_g.get_relation_record_names(inst_name)
            relation_trg = self.trg.sub_graph_of_records(relation_record_names)
            node2priority = self.inherit_g.get_node_with_priority(inst_name)

            # c. 找到relation_trg中的所有起始节点
            init_nodes = relation_trg.get_start_use_locations()

            flow_type_dict = defaultdict(list)  # {start_node: [(end_node, end_node_type, end_edge), ...]}
            for start_node in init_nodes:
                # d. 提取流：在子图TRG中提取起始节点出发的所有流向构成的子图DataGraph
                dfg = relation_trg.sub_graph_of_start_node(start_node)
                # e. 终止节点三种类型的判定
                for end_edge in dfg.get_attr_end_edges():
                # for end_edge in dfg.get_end_edges():
                    _, end_node, _ = end_edge
                    flow_type_dict[start_node].append((end_node, dfg.get_end_node_type(end_edge), end_edge))

            # f. name_value_map 是一对一映射，attribute类型终止节点的个数：
            attr_edge_res = defaultdict(list)   # {start_node: [end_node]}
            # f. 1) 留下所有end_node_type == ATTRIBUTE的start_node, end_node对
            for start_node, end_info in flow_type_dict.items():
                for end_node, end_node_type, end_edge in end_info:  # [(end_node, end_node_type), ...]
                    if end_node_type is EndNodeType.ATTRIBUTE:
                        attr_edge_res[start_node].append(end_node)

            # f. 2) 一般但单进多出不出现
            for start_node, end_nodes in list(attr_edge_res.items()):
                end_nodes = tmp_remove_bits_edges(end_nodes, self.dataset)
                if len(end_nodes) == 0:
                    pass
                else:
                    assert len(end_nodes) == 1, f"{end_nodes}, " \
                                                            f"single in multiple outs found."
                attr_edge_res[start_node] = end_nodes

            attr_name_res = defaultdict(list)  # {attr_name: [(start_node, end_node)...]}
            for start_node, end_nodes in attr_edge_res.items():
                if len(end_nodes) == 0:
                    continue
                name = self.dataset.get_tokens_by_loc(end_nodes[0])
                attr_name_res["".join(name)].append((start_node, end_nodes[0]))  # {Size: (4.loc, Size.loc)}

            # f. 3) 多进单出 / 多进多出,通过优先级选出start_node
            priority_filter_res = defaultdict(list)
            for attr_name, edges in attr_name_res.items():
                prefer_start_node, prefer_end_node = edges[0]
                for start_node, end_node in edges[1:]:
                    # start_node > prefer_start_node
                    if self.inherit_g.priority_compare(
                            start_node, end_node, prefer_start_node, prefer_end_node, node2priority):
                        prefer_start_node, prefer_end_node = start_node, end_node
                attr_name_res[attr_name] = [(prefer_start_node, prefer_end_node)]

                # g. 1) 收集name_value_map通过优先级过滤掉的属性, (start_node, [value_loc1, value_loc2])
                for start_node, end_node in edges:
                    if end_node != prefer_end_node:
                        priority_filter_res[attr_name].append((start_node, end_node))

            # g. 2) 收集end_node_type != ATTRIBUTE的节点, 记录类型， (start_node, (next_node, type))
            non_attr_type_res = defaultdict(list)  # {start_node: [(next_node, end_nodes, type_list, )]....}
            for start_node, end_info in flow_type_dict.items():
                if start_node not in attr_edge_res:
                    # 不在attr_edge_res中，也就是所有end_note_type都不是ATTRIBUTE
                    for end_node, end_node_type, end_edge in end_info:  # [(end_node, end_node_type, end_edge), ...]
                        assert end_node_type is not EndNodeType.ATTRIBUTE

                    edges = list(relation_trg.edges[start_node].items())
                    assert len(edges) == 1, f"{edges}filter first edge multiple outs!"  # 只有变量才能被引用多次，value只会有一个流向
                    next_node, _ = edges[0]
                    type_loc = get_type_loc_by_var_loc(next_node, self.dataset)
                    type_str = self.dataset.get_tokens_by_loc(type_loc)
                    # non_attr_res[start_node].append((next_node, type_list))
                    non_attr_type_res["".join(type_str)].append((start_node, next_node))

            name_value_map[inst_name] = attr_name_res
            filter_name_value_map[inst_name] = priority_filter_res
            filter_non_attr_map[inst_name] = non_attr_type_res
        return {
            "name_value_map": name_value_map,
            "filter_name_value_map": filter_name_value_map,
            "filter_non_attr_map": filter_non_attr_map
        }

    def check(self):
        pass
