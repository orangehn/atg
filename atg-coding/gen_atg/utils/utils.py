from parser_util_v2.dataset_constructor import Dataset
from parser_util_v2.data_flow_v2 import GraphHandler
from gen_atg.utils.ast import AstTransform
from gen_atg.utils.adjust_reference_record import RecordAdjustor
from copy import deepcopy

"""
需求：
1. 继承顺序
2. name value 替换:
    - 最简单的k v替换，只要通过TRG找到每个name对应的value的位置，替换即可
    - 当一个name对应多出value的时候，
    - 当一个value对应多个name的时候，
    - 当value不是基础类型，而是一个dag、list的时候(supertoken) value应该给一个location的范围，而不是一个Location
    - 当value是bits类型的时候
3. 找到每个参数的赋值的位置，(传参的位置)

现在的结构：
dataset的初始结构{filename: [[token1, tokne2, ...]]}
dataset建立的映射结构{record_name: record ...}
dataset中record是个半AST结构，data属性存储了解析出来的name, args, supers, content等属性

通过 location 来对 dataset 进行索引(filename, record_id, token_id, length) 来定位dataset的任意连续片段, line_id用于debug
    self.filename = filename
    self.record_id = record_id
    self.token_id = token_id
    self.length = length
    self.line_id = line_id
    
使用的优点是：
    - 可以很容易对dataset进行遍历(for)
    - 可以通过location方便地记录dataset中的任何value的location
    - 冗余存储，record中的半AST结构data很方便获得解析后的信息，读取方便
缺点：
    - 修改困难，根据location修改value，如果长度不同，则会使的其他location位置出错
    - 存在data(半AST) 和 数组两种存储形式，修改了一种，另一种也需要更新，冗余存储带来冗余修改
    

AST如何定位location
"""

"""
dataset => AST:
    record => AST
"""


class ReferenceCode(object):
    def __init__(self, reference_td_dir):
        self.reference_td_dir = reference_td_dir

        self.relation_record_names = None
        self.asts = None
        self.name_value_map = None

    def build(self, inst_names):
        dataset = Dataset(self.reference_td_dir)  # Syntax Trees Maintain
        # r = dataset.records_map['MicroMipsInstBase']

        # 获得name_value_map, 以及需要删除设置默认值的位置
        G = GraphHandler(dataset)  # Token Reference Graph Maintain
        G.build_graph()
        G.check()
        graph_results = G.run(inst_names)
        relation_record_names = G.get_relation_sets(inst_names)

        # 转换成ast方便修改
        ast_transform = AstTransform(dataset)
        asts = ast_transform.record_to_ast()
        ast_results = ast_transform.locations_to_asts(graph_results, asts)

        # 调整模板, 删除部分内容并设置默认值
        record_adjustor = RecordAdjustor(asts)
        asts = record_adjustor.run(ast_results)
        ast_results = AstTransform.asts_to_ids(ast_results, asts)

        self.relation_record_names = relation_record_names
        self.asts = asts
        self.name_value_map = ast_results["name_value_map"]

    def get_reference_code(self, inst_name):
        relation_names = self.relation_record_names[inst_name]
        relation_asts = {}
        for record_name in relation_names:
            relation_asts[record_name] = self.asts[record_name]
        relation_asts = deepcopy(relation_asts)
        name_value_map = self.name_value_map[inst_name]

        ast_results = {"name_value_map": {inst_name: name_value_map}}
        ast_results = AstTransform.ids_to_asts(ast_results, relation_asts)
        name_value_map = ast_results["name_value_map"][inst_name]
        return relation_asts, name_value_map

    def replace_with_kv_and_rename(self, relation_asts, name_value_map, input_inst_attrs, refer_inst_name, new_inst_name):
        record_adjustor = RecordAdjustor(relation_asts)
        relation_asts, new_input_inst_attrs = record_adjustor.set_attr_to_input_value(name_value_map, input_inst_attrs)

        # rename like: def ADD_MIPS => def ADD_RISCV
        relation_asts[refer_inst_name].rename_class_name(new_inst_name)
        relation_asts[new_inst_name] = relation_asts.pop(refer_inst_name)
        return relation_asts, new_input_inst_attrs


class CodeMerger(object):
    def __init__(self):
        pass

    def is_same_str(self, record_asts):
        assert len(record_asts) > 0
        if len(record_asts) == 1:
            return True
        ast_str = str(record_asts[0])
        for ast in record_asts[1:]:
            if ast_str != str(ast):
                return False
        return True

    def run(self, gen_codes):
        """
            gen_codes: {
                inst_name:{
                    relation_record_name1: record_ast1
                }
            }
        """
        new_gen_codes = []
        # record_group_by_name = defaultdict(list)
        # for inst_name, relation_codes in gen_codes.items():
        #     for record_name, record_ast in relation_codes.items():
        #         record_group_by_name[record_name].append(record_ast)
                # new_gen_codes.append(record_ast)

        # same_record_name = set()
        # for record_name, record_asts in record_group_by_name.items():
        #     if self.is_same_str(record_asts):
        #         same_record_name.add(record_name)

        for inst_name, relation_codes in gen_codes.items():
            relation_record_names = set(relation_codes.keys())
            assert inst_name in relation_codes, inst_name
            for record_name, record_ast in relation_codes.items():
                if record_name != inst_name:
                    record_ast.add_suffix_to_class_name(suffix=f"_{inst_name}")
                else:
                    assert str(record_ast.data) == 'def', record_ast
                record_ast.rename_super_classes(suffix=f"_{inst_name}", replacing_names=relation_record_names)
                new_gen_codes.append(record_ast)
        return new_gen_codes
