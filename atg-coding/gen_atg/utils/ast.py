from parser_util_v2.dataset_type import SuperToken, Token
from collections import OrderedDict
from copy import deepcopy


class Node(object):
    def __init__(self, children=None, data=None):
        self.children = children
        self.data = data
        if self.children is not None:
            for child_name, child_node in self.children.items():
                assert isinstance(child_node, Node), f"{type(self)}: {child_name} is excepted as a Node," \
                                                     f" but got {type(child_node)}({child_node})."

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "not implement"

    def traversal(self, callback):
        """
            遍历ast
        """
        ret = [callback(self)]

        if self.children is None:
            return []
        elif isinstance(self.children, dict):
            for child_name, child_node in self.children.items():
                if child_node is not None:
                    ret.extend(child_node.traversal(callback))
        elif isinstance(self.children, list):
            for child_node in self.children:
                if child_node is not None:
                    ret.extend(child_node.traversal(callback))
        else:
            raise TypeError()
        return ret

    def update(self, data):
        raise NotImplementedError

    def get_start_loc(self):
        def get_loc(token):
            if hasattr(token, 'loc'):
                return token.loc
            return None

        if isinstance(self, TokenNode):
            return get_loc(self.data)
        elif isinstance(self, ListNode):
            return get_loc(self.children[0].data)
        else:
            raise TypeError(type(self))


class TokenNode(Node):
    def __init__(self, token):
        super(TokenNode, self).__init__(None, token)
        # if self.data == 'Predicate':
        #     print("__init__", id(self.data), self.data, self.data.loc)

    def to_list(self):
        return [self.data]

    def __str__(self):
        # if self.data == 'Predicate':
        #     print("__str__", id(self.data), self.data, self.data.loc)
        return self.data

    def update(self, data):
        self.data = data


class ListNode(Node):
    """
        1. identifier token: ADD
        2. string          : "add" / "$rd = $rs"
        3. int             : 6
        4. bit             : 0/1/0b0/0b1
        5. bits            : 0b0111 / 7
        6. dag             : (outs) / (OpNode RO:$rs, RO:$rt) / (set RO:$rd, (OpNode RO:$rs, RO:$rt))
        7. null            : null_frag /
        8. list            : [] / [xxx, ]

        9. string拼接: opstr # "\t$rd, $rs, $rt" or  !strconcat(opstr,"\t$rd, $rs, $rt")
        10. 属性引用   ：Form.value
    """
    def __init__(self, nodes, data=None):
        self.children = nodes
        self.data = data
        for child_id, child_node in enumerate(self.children):
            assert isinstance(child_node, Node), f"{type(self)}: {child_id} is excepted as a Node," \
                                                 f" but got {type(child_node)}({child_node})."

    def to_list(self):
        ret = []
        for node in self.children:
            ret.extend(node.to_list())
        return ret

    def __str__(self):
        ret = []
        for node in self.children:
            ret.append(str(node))
        new_ret = [ret[0]]
        for i in range(len(ret) - 1):
            if ret[i].isidentifier() and ret[i + 1].isidentifier():
                new_ret.append(" ")
            new_ret.append(ret[i + 1])
        return "".join(new_ret)

    def update(self, children_data: (str, list)):
        if isinstance(children_data, str):
            children_data = [children_data]

        if len(self.children) == 1:
            assert len(children_data) == 1
            self.children[0].update(children_data[0])
        else:
            tokens = []
            for data in children_data:
                assert isinstance(data, str)
                tokens.append(TokenNode(data))
            self.children = tokens


# SuperTokenNode = ListNode

class ArgNode(Node):
    def __init__(self, type, name, value=None, data=None):
        self.children = OrderedDict(
            type=type, name=name, value=value
        )
        self.data = data
        for child_name, child_node in self.children.items():
            if child_name == 'value' and child_node is None:
                continue
            assert isinstance(child_node, Node), f"{type(self)}: {child_name} is excepted as a Node," \
                                                 f" but got {type(child_node)}({child_node})."

    def to_list(self):
        value_list = []
        if self.children['value'] is not None:
            value_list = ["="] + self.children['value'].to_list()
        return self.children['type'].to_list() + self.children['name'].to_list() + value_list

    def __str__(self):
        value_str = ""
        if self.children['value'] is not None:
            value_str = "=" + str(self.children['value'])
        return f"{self.children['type']}  {self.children['name']}{value_str}"


class SuperClassNode(Node):
    def __init__(self, name: TokenNode, args: ListNode, data=None):
        children = OrderedDict(
            name=name, args=args
        )
        super(SuperClassNode, self).__init__(children, data)

    def to_list(self):
        arg_list = self.children["args"].to_list()
        if len(arg_list) > 0:
            arg_list = ['<'] + arg_list + ['>']
        return self.children['name'].to_list() + arg_list

    def __str__(self):
        arg_str = str(self.children["args"])
        if len(arg_str) > 0:
            arg_str = f"<{arg_str}>"
        return f"{self.children['name']} {arg_str}"


class CommaListNode(ListNode):
    def __init__(self, nodes, data=None):
        super(CommaListNode, self).__init__(nodes, data)

    def to_list(self):
        ret = []
        for i, node in enumerate(self.children):
            ret.extend(node.to_list())
            if i != len(self.children) - 1:
                ret.append(',')
        return ret

    def __str__(self):
        ret = []
        for i, node in enumerate(self.children):
            ret.append(str(node))
        return ", ".join(ret)


StmtNode = ArgNode


class ContentListNode(ListNode):
    def __init__(self, nodes, data=None):
        super(ContentListNode, self).__init__(nodes, data)

    def to_list(self):
        ret = []
        for node in self.children:
            ret.extend(node.to_list() + [';'])
        return ret

    def __str__(self):
        ret = []
        for i, node in enumerate(self.children):
            ret.append(f"\t{str(node)};\n")
        return "".join(ret)


class RecordNode(Node):
    def __init__(self, name, args, super_classes, contents, data):
        children = OrderedDict(
            name=name, args=args, super_classes=super_classes, contents=contents
        )
        super(RecordNode, self).__init__(children, data)
        self.id2node = {}

    def fresh_id2node(self):
        id2node = {}

        def set_id(node):
            node.id = (str(self.children['name']), len(id2node))
            id2node[len(id2node)] = node

        self.traversal(set_id)
        self.id2node = id2node

    def append_super_class(self, super_class_name):
        super_classes_node = self.children['super_classes']  # CommonListNode
        super_class_node = SuperClassNode(TokenNode(super_class_name), CommaListNode([]))
        super_classes_node.children.append(super_class_node)

    def append_content_str(self, body_str):
        assert body_str[-1] == ';', body_str
        contents_node = self.children['contents']  # CommonListNode
        body_node = StmtNode(type=TokenNode(""), name=TokenNode(body_str[:-1]))
        contents_node.children.append(body_node)

    def rename_class_name(self, new_name):
        self.children["name"].update(new_name)

    def add_suffix_to_class_name(self, suffix):
        name_str = str(self.children["name"])
        self.children["name"].update(name_str + suffix)

    def rename_super_classes(self, suffix, replacing_names):
        super_classes_node = self.children['super_classes']  # CommonListNode
        for super_node in super_classes_node.children:
            class_name = str(super_node.children['name'])
            if class_name in replacing_names:
                super_node.children['name'].update(class_name + suffix)

    def to_list(self):
        arg_list = self.children['args'].to_list()
        if len(arg_list) > 0:
            arg_list = ['<'] + arg_list + ['>']
        super_list = self.children["super_classes"].to_list()
        if len(super_list) > 0:
            super_list = [':'] + super_list
        content_list = self.children["contents"].to_list()
        if len(content_list) > 0:
            content_list = ['{'] + content_list + ['}']
        else:
            content_list = [';']
        return self.data.to_list() + self.children["name"].to_list() + arg_list + super_list + content_list

    def __str__(self):
        arg_str = str(self.children['args'])
        if len(arg_str) > 0:
            arg_str = f"<{arg_str}>"
        super_str = str(self.children["super_classes"])
        if len(super_str) > 0:
            super_str = f" : {super_str}"
        content_str = str(self.children["contents"])
        if len(content_str) > 0:
            content_str = f"{{\n{content_str}}}"
        else:
            content_str = ";"
        return f"{self.data} {self.children['name']}{arg_str}{super_str}{content_str}"


# Ast Transform Tools #########################################

from parser_util_v2.dataset_constructor import Location


def _exp_to_ast(tokens):
    """list:[token, list, ...] => [TokenNode(token), TokenNode(token) .....]"""
    if isinstance(tokens, Token):
        return [TokenNode(tokens)]
    assert isinstance(tokens, list), tokens
    token_node = []
    for token in tokens:
        if isinstance(token, Token):
            token_node.append(TokenNode(token))
        elif isinstance(token, list):
            token_node.extend(_exp_to_ast(token))
        else:
            raise TypeError(f"{token} ({type(token)})")
    return token_node


def exp_to_ast(tokens):
    return ListNode(_exp_to_ast(tokens))


def to_ast(data, stmt_type_token):
    name_node = exp_to_ast(data["name"])

    arg_nodes = []
    args = data["args"]
    for arg in args:
        type_node = exp_to_ast(arg[0])  # (['bits', ['<', '3', '>']], ['major'], None)
        arg_name_node = exp_to_ast(arg[1])
        arg_value_node = None if arg[2] is None else exp_to_ast(arg[2])
        arg_nodes.append(ArgNode(type_node, arg_name_node, arg_value_node))

    super_class_nodes = []
    super_classes = data["supers"]
    for super in super_classes:
        s_name_node = exp_to_ast(super[0])
        s_arg_nodes = []  # <a, a # b> # [['(', 'outs', 'RO', ':', '$', 'rd', ')']]
        for s_arg in super[1]:
            s_arg_nodes.append(exp_to_ast(s_arg))
        super_class_nodes.append(SuperClassNode(s_name_node, CommaListNode(s_arg_nodes)))

    content_nodes = []
    contents = data["content"]
    for i, content in enumerate(contents): # ['bits', ['<', '5', '>']]
        content_type_node = content[0]
        if len(content_type_node) == 0 and data["content_let"][i]:
            content_type_node = [Token('let')]
        content_type_node = exp_to_ast(content_type_node)
        content_name_node = exp_to_ast(content[1])
        content_value_node = None if content[2] is None else exp_to_ast(content[2])
        content_nodes.append(StmtNode(content_type_node, content_name_node, content_value_node))

    r_ast = RecordNode(
        name=name_node,
        args=CommaListNode(arg_nodes),
        super_classes=CommaListNode(super_class_nodes),
        contents=ContentListNode(content_nodes),
        data=TokenNode(stmt_type_token)
    )
    # print(r_ast)

    return r_ast


def traversal_ast(ast: Node, loc2ast):
    """
        遍历ast
    """
    loc = None
    if isinstance(ast, TokenNode):
        loc = ast.data.loc
    elif isinstance(ast, ListNode) and (not isinstance(ast, (CommaListNode, ContentListNode))) \
            and len(ast.children) > 1:
        assert isinstance(ast.children[0], TokenNode), ast.children[0]
        assert isinstance(ast.children[-1], TokenNode), ast.children[-1]
        loc_start, loc_end = ast.children[0].data.loc, ast.children[-1].data.loc
        assert loc_start is not None and loc_end is not None, (ast, loc_start, loc_end)
        loc = Location.create_location(loc_start.filename, loc_start.record_id, loc_start.token_id,
                                       length=loc_end.token_id - loc_start.token_id + 1)
    if loc is not None:
        assert loc not in loc2ast, (loc, loc2ast[loc])
        loc2ast[loc] = ast
    # print("xxxx", type(ast))

    if ast.children is None:
        return
    elif isinstance(ast.children, dict):
        for child_name, child_node in ast.children.items():
            if child_node is not None:
                traversal_ast(child_node, loc2ast)
    elif isinstance(ast.children, list):
        for child_node in ast.children:
            if child_node is not None:
                traversal_ast(child_node, loc2ast)
    else:
        raise TypeError()


class Loc2AstCallBack(object):
    def __init__(self):
        self.loc2ast = {}

    def __call__(self, ast: Node):
        loc = None
        if isinstance(ast, TokenNode):
            loc = ast
        elif isinstance(ast, ListNode) and (not isinstance(ast, (CommaListNode, ContentListNode))) \
                 and len(ast.children) > 1:
            assert isinstance(ast.children[0], TokenNode), ast.children[0]
            assert isinstance(ast.children[-1], TokenNode), ast.children[-1]
            loc_start, loc_end = ast.children[0].data.loc, ast.children[-1].data.loc
            assert loc_start is not None and loc_end is not None, (ast, loc_start, loc_end)
            loc = Location.create_location(loc_start.filename, loc_start.record_id, loc_start.token_id,
                                           length=loc_end.token_id - loc_start.token_id + 1)
        if loc is not None:
            assert loc not in self.loc2ast, loc
            self.loc2ast[loc] = ast


def name_value_loc_to_ast(name_value_loc, loc2ast, dataset):
    """
    name_value_loc: {
        inst_name:{
            attr_name: [(prefer_start_node, prefer_end_node), ...]
        }
    }
    """
    # for loc, ast in loc2ast.items():
    #     if dataset.get_tokens_by_loc(loc)[0] == "1":
    #         print(loc, dataset.get_tokens_by_loc(loc), ':', ast, id(loc))

    # 2. map location in name_value_map to ast
    name_value_ast = {}
    for record_name in name_value_loc:
        name_value_loc_r = name_value_loc[record_name]
        name_value_ast_r = {}
        for attr_name, loc_infos in name_value_loc_r.items():
            ast_infos = []
            # assert len(loc_info) == 1, loc_info
            for loc_info in loc_infos:
                ast_info = []
                for loc in loc_info:
                    assert isinstance(loc, Location), loc
                    # print(loc, id(loc))
                    assert loc in loc2ast, dataset.get_tokens_by_loc(loc)
                    ast_info.append(loc2ast[loc])
                ast_infos.append(tuple(ast_info))
            name_value_ast_r[attr_name] = ast_infos
        name_value_ast[record_name] = name_value_ast_r
    return name_value_ast


class AstTransform(object):
    """
        1） 将record转成ast
        2） name_value_map中的location索引转成ast索引
    """
    def __init__(self, dataset):
        self.dataset = dataset  # should be read only or debug only

    def record_to_ast(self):
        record_names = list(self.dataset.records_map.keys())
        asts = {}
        for record_name in record_names:
            record = self.dataset.records_map[record_name]
            ast = to_ast(record.data, record[0])
            ast.fresh_id2node()  # set id and set id2node map
            asts[record_name] = ast
        return asts

    # def locations_to_asts(self, graph_result, asts):
    #     """
    #     name_value_map, filter_name_value_map, filter_non_attr_map = maps
    #     name_value_map/filter_name_value_map : {
    #         inst_name:{
    #             attr_name: [(prefer_start_node, prefer_end_node), ...]
    #         }
    #     }
    #     filter_non_attr_map : {
    #         inst_name:{
    #             type_str: [(start_node, end_node), ...]
    #         }
    #     }
    #     """
    #     loc2ast = {}
    #     # 1. get loc2ast
    #     # loc2ast_callback = Loc2AstCallBack()
    #     for record_name in self.dataset.records_map:
    #         # print(f"record  {record_name}-----------\n")
    #         # print(asts[record_name])
    #         # print(f" finished record {record_name} --------------------\n")
    #         traversal_ast(asts[record_name], loc2ast)
    #         # ast_r.traversal(callback=loc2ast_callback)
    #     # for record_name, record in dataset.records_map.items():
    #     #     print(f"record  {record_name}-----------\n")
    #     #     print(asts[record_name])
    #     #     print(f" finished record {record_name} --------------------\n")
    #
    #     # loc2ast = loc2ast_callback.loc2ast
    #
    #     name_value_map = name_value_loc_to_ast(graph_result["name_value_map"], loc2ast, self.dataset)
    #     filter_name_value_map = name_value_loc_to_ast(graph_result["filter_name_value_map"], loc2ast, self.dataset)
    #     filter_non_attr_map = name_value_loc_to_ast(graph_result["filter_non_attr_map"], loc2ast, self.dataset)
    #     return {
    #         "name_value_map": name_value_map,
    #         "filter_name_value_map": filter_name_value_map,
    #         "filter_non_attr_map": filter_non_attr_map
    #     }

    def locations_to_asts(self, graph_result, asts):
        """
        name_value_map, filter_name_value_map, filter_non_attr_map = maps
        name_value_map/filter_name_value_map : {
            inst_name:{
                attr_name: [(prefer_start_node, prefer_end_node), ...]
            }
        }
        filter_non_attr_map : {
            inst_name:{
                type_str: [(start_node, end_node), ...]
            }
        }
        """
        loc2ast = {}
        # 1. get loc2ast
        # loc2ast_callback = Loc2AstCallBack()
        for record_name in self.dataset.records_map:
            # print(f"record  {record_name}-----------\n")
            # print(asts[record_name])
            # print(f" finished record {record_name} --------------------\n")
            traversal_ast(asts[record_name], loc2ast)
            # ast_r.traversal(callback=loc2ast_callback)
        # for record_name, record in dataset.records_map.items():
        #     print(f"record  {record_name}-----------\n")
        #     print(asts[record_name])
        #     print(f" finished record {record_name} --------------------\n")

        # loc2ast = loc2ast_callback.loc2ast

        def transform(loc):
            assert isinstance(loc, Location), loc
            # print(loc, id(loc))
            assert loc in loc2ast, self.dataset.get_tokens_by_loc(loc)
            return loc2ast[loc]

        for key in ["name_value_map", "filter_name_value_map", "filter_non_attr_map"]:
            graph_result[key] = AstTransform.traversal_transform_map(graph_result[key], transform)
        return graph_result

    @staticmethod
    def asts_to_ids(ast_result, asts):
        # for record_name, ast in asts.items():
        #     ast.fresh_id2node()

        def transform(ast):
            assert isinstance(ast, Node)
            return ast.id

        for key in ["name_value_map", "filter_name_value_map", "filter_non_attr_map"]:
            ast_result[key] = AstTransform.traversal_transform_map(ast_result[key], transform)
        return ast_result

    @staticmethod
    def ids_to_asts(ast_result, asts):
        def transform(ast_id):
            record_name, node_id = ast_id
            return asts[record_name].id2node[node_id]

        for key in ["name_value_map", "filter_name_value_map", "filter_non_attr_map"]:
            if key in ast_result:
                ast_result[key] = AstTransform.traversal_transform_map(ast_result[key], transform)
        return ast_result

    @staticmethod
    def traversal_transform_map(name_value_map, transform):
        """
        name_value_loc: {
            inst_name:{
                attr_name: [(prefer_start_node, prefer_end_node), ...]
            }
        }
        """
        # for loc, ast in loc2ast.items():
        #     if dataset.get_tokens_by_loc(loc)[0] == "1":
        #         print(loc, dataset.get_tokens_by_loc(loc), ':', ast, id(loc))

        # 2. map location in name_value_map to ast
        name_value_ast = {}
        for record_name in name_value_map:
            name_value_loc_r = name_value_map[record_name]
            name_value_ast_r = {}
            for attr_name, loc_infos in name_value_loc_r.items():
                ast_infos = []
                # assert len(loc_info) == 1, loc_info
                for loc_info in loc_infos:
                    ast_info = []
                    for loc in loc_info:
                        ast_info.append(transform(loc))
                    ast_infos.append(tuple(ast_info))
                name_value_ast_r[attr_name] = ast_infos
            name_value_ast[record_name] = name_value_ast_r
        return name_value_ast


if __name__ == '__main__':
    """
    我们以一条record为例子，建一个AST:
    class A<int arg1, string opcode>: B<arg1, 0>, C{
        int length;
        int Size = 0;
        let Name = 'xxx';
        let Opcode = opcode;
    }

    class A<int arg1, string opcode>: B<arg1, 0>, C{
        bits<7> Opcode = opcode;
    }
    """
    arg = ArgNode(TokenNode("int"), TokenNode("arg1"))
    r = RecordNode(
        name=TokenNode("A"),
        args=CommaListNode([
            arg,
            ArgNode(ListNode([TokenNode("bits"), TokenNode("<"), TokenNode("7"), TokenNode(">")]),
                    TokenNode("arg2"))
        ]),
        super_classes=CommaListNode([
            SuperClassNode(TokenNode("B"), CommaListNode([TokenNode("arg1"), TokenNode("0")])),
            SuperClassNode(TokenNode("C"), CommaListNode([]))
        ]),
        contents=ContentListNode([
            StmtNode(ListNode([TokenNode("bits"), TokenNode("<"), TokenNode("7"), TokenNode(">")]),
                     TokenNode("Opcode"), TokenNode("Opcode")),
            StmtNode(ListNode([TokenNode("bits"), TokenNode("<"), TokenNode("5"), TokenNode(">")]),
                     TokenNode("Opcode"), TokenNode("Opcode")),

            StmtNode(TokenNode("dag"), TokenNode("Out"), ListNode([
                TokenNode("("), TokenNode("outs"), TokenNode("RO"), TokenNode(":"), TokenNode("$"), TokenNode("rd"),
                TokenNode(")")]))
        ]),
        data=TokenNode("class")
    )

    print(r)
    r.append_supper_class("E")
    print(r)

    # print(r.to_list())
    # print(r)
    r_copy = deepcopy(r)
    arg.children["type"].update("float")
    print(r)
    print(r_copy)

    """
    name_value_loc_map => name_
    """
    config_dict = {
        # RV64
        # "test": {
        #     "td": "./Input/Instruction",
        #     "input_files": ["./Input/inst_set_input/I/test.csv", "./Input/inst_set_input/I/test_inst_match.csv", ""],
        #     "generate_func": generation_process,
        #     "out": "./output/inst_set/test.td"
        # },
        # "M": {
        #     "td": "./Input/Instruction",
        #     "input_files": ["./Input/inst_set_input/m/m.csv", "./Input/inst_set_input/m/m_inst_match.csv", ""],
        #     "generate_func": generation_process,
        #     "out": "./output/inst_set/m.td"
        # },
        "I": {
            "td": "./Input/Instruction",
            # Input/inst_set_input/{inst_set}/{inst_set}_from_json.csv
            "input_files": ["./Input/inst_set_input/I/I_from_json.csv", "./Input/inst_set_input/I/I_inst_match.csv",
                            ""],
            "out": "./output/inst_set/I.td"
        },
    }
