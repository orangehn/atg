"""
1.	替换部分代码，参考平台代码对存在依赖，如果使用include的方式导入这些类定义，又可能和新平台的类定义存在重名的冲突。（InstFormat, …）
这些依赖都是些平台自定义的非基础类型。
=》非基础类型替换为默认类型，非基础类型的值设置为默认值。
建立两个dict，分别用于类型替换和值替换。
=》dfg => start_loc => ast => 需要替换的类型 => 进行值替换 {type: default_value}
=》遍历record ast，找到所有形参和stmt里面的type=>非基础类型？（InstFormat, 寄存器）
=>进行类型替换 {type: new_type}

2.	多余属性设置默认值
name_value_map中多余（每个）属性设置默认值 {name: default_value}

3.	删除部分代码：
bits（不是简单的kv形式，而是索引，key, value三元形式），不能使用kv替换生成，使用模板生成，但是bits的长度不同会和模板定义冲突。


实际代码实现：
a. filter_name_value_map, 根绝 attr_default_value_dict 将指定属性的start_loc设置为默认值
b. filter_non_attr_map，根据 type_default_value_dict 将指定类型的start_loc设置为默认值
c. 遍历整个dataset里的record，根据 type_default_type_dict 将所有形参和stmt里面指定类型替换成默认类型
"""

attr_default_value_dict = dict(
    # InstructionEncoding
    hasCompleteDecoder="true", Size="0", DecoderNamespace='""', Predicates='[]', DecoderMethod='""',
    # Instruction: InstructionEncoding
    Namespace='""', AsmString='""', Uses='[]', Defs='[]', CodeSize='0', AddedComplexity='0',
    isPreISelOpcode='false', isReturn='false', isBranch='false', isEHScopeReturn='false', isIndirectBranch='false',
    isCompare='false', isMoveImm='false', isMoveReg='false', isBitcast='false', isSelect='false', isBarrier='false',
    isCall='false', isAdd='false', isTrap='false', canFoldAsLoad='false', mayLoad='?', mayStore='?',
    mayRaiseFPException='false', isConvertibleToThreeAddress='false', isCommutable='false', isTerminator='false',
    isReMaterializable='false', isPredicable='false', isUnpredicable='false', hasDelaySlot='false',
    usesCustomInserter='false', hasPostISelHook='false', hasCtrlDep='false', isNotDuplicable='false',
    isConvergent='false', isAuthenticated='false', isAsCheapAsAMove='false', hasExtraSrcRegAllocReq='false',
    hasExtraDefRegAllocReq='false', isRegSequence='false', isPseudo='false', isExtractSubreg='false',
    isInsertSubreg='false', variadicOpsAreDefs='false', hasSideEffects='?', isCodeGenOnly='false',
    isAsmParserOnly='false', hasNoSchedulingInfo='false', Itinerary='NoItinerary', Constraints='""',
    DisableEncoding='""', PostEncoderMethod='""', TSFlags='0', AsmMatchConverter='""', TwoOperandAliasConstraint='""',
    AsmVariantName='""', UseNamedOperandTable='false', FastISelShouldIgnore='false',

    #
    InOperandList='(ins)', OutOperandList='(outs)', Pattern='[]', Opcode='0',
    isCTI='0', Arch='""', BaseOpcode='""',
    HardFloatPredicate='[]', GPRPredicates='[]', EncodingPredicates='[]', PTRPredicates='[]',
    SYMPredicates='[]', FGRPredicates='[]', InsnPredicates='[]', ASEPredicate='[]', AdditionalPredicates='[]',
    SoftFail='0', FormBits='0', Value='0',

    OpNode='null_frag',

    # MIPS
    IsPCRelativeLoad='0', hasFCCRegOperand='0', hasForbiddenSlot='0',
    hasUnModeledSideEffects='1',

    opstr='""',

    # 引入RISCV平台类别
    Form='InstFormatR',

    # ppc
    # PPC970_First='0', PPC970_Single='0', PPC970_Cracked='0', PPC970_Unit='0', XFormMemOp='0', BaseName="",
)
ignore_attrs = {
    # mips bits
    "offset", "base", "fcc", 'Addr', 'rs', "Opcode",
    "IsPCRelativeLoad", "hasFCCRegOperand", "hasForbiddenSlot", "hasUnModeledSideEffects", "isCTI",
    # arc bits
    "S9", "B", "LImmReg", "LImm",
    # ppc bits
    "Interpretation64Bit", "PPC64", "RC", "RB", "FRC", "FRB", "FRA", "BIBO", "CR", "BO", "BI", "BH", "A",
    "L", "C", "RST", "CRA", "CRB", "CRD",
    "PPC970_First", "PPC970_Single", "PPC970_Cracked", "PPC970_Unit", "XFormMemOp", "BaseName"
}
bit_encoding_attr = [
    'frag_name', 'start_bit', 'end_bit', 'value', 'is_op', 'op_type', 'Mode', 'is_signed', 'NoZero', 'ILLEGAL', 'LSB'
]
filter_attr = [
    'BaseOpcode', 'OpNode'
]
fix_attr = {
    'GPRPredicates':None, 'EncodingPredicates':None, 'ASEPredicate':None, 'HardFloatPredicate':None, 'InsnPredicates':None,
    'AdditionalPredicates':None, 'PTRPredicates':None, 'FGRPredicates':None, "Itinerary":None, "DecoderMethod":None,
    'DisableEncoding':None, 'Pattern':None, 'AsmMatchConverter':None
}
tsp_norm_dict = dict(
    opstr="rname", Form="Format", FormBits="Set"
)
instr_dict = dict(
    # InstructionEncoding
    hasCompleteDecoder="true", Size="0", DecoderNamespace='""', Predicates='[]', DecoderMethod='""',
    # Instruction: InstructionEncoding
    Namespace='""', AsmString='""', Uses='[]', Defs='[]', CodeSize='0', AddedComplexity='0',
    isPreISelOpcode='false', isReturn='false', isBranch='false', isEHScopeReturn='false', isIndirectBranch='false',
    isCompare='false', isMoveImm='false', isMoveReg='false', isBitcast='false', isSelect='false', isBarrier='false',
    isCall='false', isAdd='false', isTrap='false', canFoldAsLoad='false', mayLoad='?', mayStore='?',
    mayRaiseFPException='false', isConvertibleToThreeAddress='false', isCommutable='false', isTerminator='false',
    isReMaterializable='false', isPredicable='false', isUnpredicable='false', hasDelaySlot='false',
    usesCustomInserter='false', hasPostISelHook='false', hasCtrlDep='false', isNotDuplicable='false',
    isConvergent='false', isAuthenticated='false', isAsCheapAsAMove='false', hasExtraSrcRegAllocReq='false',
    hasExtraDefRegAllocReq='false', isRegSequence='false', isPseudo='false', isExtractSubreg='false',
    isInsertSubreg='false', variadicOpsAreDefs='false', hasSideEffects='?', isCodeGenOnly='false',
    isAsmParserOnly='false', hasNoSchedulingInfo='false', Itinerary='NoItinerary', Constraints='""',
    DisableEncoding='""', PostEncoderMethod='""', TSFlags='0', AsmMatchConverter='""', TwoOperandAliasConstraint='""',
    AsmVariantName='""', UseNamedOperandTable='false', FastISelShouldIgnore='false',
    #
    InOperandList='(ins)', OutOperandList='(outs)', Pattern='[]',
)
base_type = {
    'string', 'int', 'dag', 'bit',  # startswith 'bits', 'list'
}
# 共有类型
common_type = {
    'Instruction', 'Predicate', 'PatFrag', 'RegisterClass', 'ValueType', 'ImmLeaf', 'SDNode', 'ComplexPattern',
    'Register', 'RegisterOperand', 'SDPatternOperator', 'Operand', 'PatLeaf', 'DAGOperand', 'ComplexPattern',
    'InstrItinClass', "SubtargetFeature", "InstrMapping", "Requires",
}
# MIPS 特定类型
quote_type = {
    # format类型
    'FIELD_CMP_FORMAT', 'FIELD_CMP_COND', 'FIELD_FMT', 'OPGROUP',
    'OPCODE2', 'OPCODE3', 'OPCODE5', 'OPCODE6',
    #
    'SplatComplexPattern'}
# 优先级最高的替换字典
type_default_value_dict = {
    'SDPatternOperator': "null_frag",
    'ComplexPattern': "addr",
    'DAGOperand': "i1imm",
}
type_default_type_dict = {
    'FIELD_CMP_FORMAT': 'Type_AUX_FM',
    'FIELD_CMP_COND': 'Type_AUX_FM',
    'FIELD_FMT': 'Type_AUX_FM',
    'OPGROUP': 'Type_AUX_FM',
    'OPCODE2': 'Type_AUX_FM',
    'OPCODE3': 'Type_AUX_FM',
    'OPCODE5': 'Type_AUX_FM',
    'OPCODE6': 'Type_AUX_FM',
    'Format': 'InstFormat',
    'SplatComplexPattern': 'ComplexPattern',
}


def get_attr_default_value(attr_name):
    if attr_name in ignore_attrs:
        return None
    assert attr_name in attr_default_value_dict, f"{attr_name} not in defalut dict."
    return attr_default_value_dict[attr_name]


def get_type_default_value(type_str):
    if type_str in type_default_value_dict:
        return type_default_value_dict[type_str]
    elif type_str.startswith('bit') or type_str in base_type:  # base type
        return None
    elif type_str.startswith("list<") and type_str.endswith('>'):
        inner_type = type_str[5:-1]
        inner_value = get_type_default_value(inner_type)
        assert inner_value is None, "inner value should be common"  # 这句可以删除
        if inner_value is None:
            return None
        else:
            return f"[{inner_value}]"
    elif type_str in common_type or type_str in quote_type:
        return "?"
    else:
        raise ValueError(type_str)


def get_type_default_type(type_str):
    if type_str in type_default_type_dict:
        return type_default_type_dict[type_str]
    if type_str in ['let']:
        return None
    if type_str in base_type or type_str in common_type or \
            type_str.startswith('bit'):  # base/common type 不替换
        return None
    if type_str.startswith('list<') and type_str.endswith(">"):
        inner_type = type_str[5:-1]
        inner_type = get_type_default_type(inner_type)
        if inner_type is None:
            return None
        else:
            return f"list<{inner_type}>"
    else:
        raise ValueError(type_str)


def in_defined_record(type_str, record_names):
    if type_str in record_names:
        return True
    elif type_str.startswith("list<") and type_str.endswith(">"):
        inner_type = type_str[5:-1]
        return in_defined_record(inner_type, record_names)
    return False


def debug_print(*args):
    # print(*args)
    pass


def filter_tsp(tsp_list):
    for i, tsp in enumerate(tsp_list):
        if tsp in filter_attr:
            del tsp_list[i]


def rename_tsp(tsp_name):
    if tsp_name in tsp_norm_dict:
        return tsp_norm_dict[tsp_name]
    else:
        return tsp_name


def normalization_tsp(tsp_list):
    norm_tsp_list = []
    for tsp in tsp_list:
        if tsp not in ignore_attrs:
            tsp = rename_tsp(tsp)
            norm_tsp_list.append(tsp)
    norm_tsp_list.extend(bit_encoding_attr)
    filter_tsp(norm_tsp_list)

    return norm_tsp_list


class RecordAdjustor(object):
    def __init__(self, asts):
        # self.dataset = dataset
        self.asts = asts

    def remove_bits(self):
        """
        bits 里面的长度不一致可能影响后面全属性模板的内容
        """
        for inst_name, ast in self.asts.items():
            # ast = RecordNode()
            # ast.children['args']
            content_list_node = ast.children['contents']
            left_stmts = []
            for stmt_node in content_list_node.children:
                type_list_node = stmt_node.children["type"]
                name_list_node = stmt_node.children['name']
                type_str = str(type_list_node)
                if type_str.startswith("bits") or type_str.startswith("field bits"):
                    continue
                elif len(name_list_node.children) > 2 and \
                        str(name_list_node.children[1]) == '{' and str(name_list_node.children[-1]) == '}':
                    continue
                left_stmts.append(stmt_node)
            content_list_node.children = left_stmts

    def set_type_to_default_type(self):
        record_names = set(self.asts.keys())
        for record_name in self.asts:
            ast = self.asts[record_name]
            for node in ast.children['args'].children + ast.children['contents'].children:
                type_node = node.children['type']
                type_str = str(type_node)
                if in_defined_record(type_str, record_names):
                    continue
                new_type_str = get_type_default_type(type_str)
                if new_type_str is not None:
                    type_node.update(new_type_str)
                    debug_print(f"{str(type_str)} -> {new_type_str}")
            for node in ast.children['super_classes'].children:
                super_name_node = node.children['name']
                type_str = str(super_name_node)
                if in_defined_record(type_str, record_names):
                    continue
                new_type_str = get_type_default_type(type_str)
                if new_type_str is not None:
                    super_name_node.update(new_type_str)
                    debug_print(f"{str(type_str)} -> {new_type_str}")

    def set_attr_to_default_value(self, filter_name_value_map, filter_non_attr_map, name_value_map):
        """
        filter_name_value_map : {
            inst_name:{
                attr_name: [(prefer_start_ast, prefer_end_ast), ...]
            }
        }
        filter_non_attr_map : {
            inst_name:{
                type_str: [(start_ast, end_ast), ...]
            }
        }
        """
        for inst_name, filter_name_value_ast in filter_name_value_map.items():
            filter_non_attr_ast = filter_non_attr_map[inst_name]
            name_value_ast = name_value_map[inst_name]

            # print(inst_name, "----------------------------------------")
            debug_print("filter_non_attr_ast ************************")
            for type_str, ast_infos in filter_non_attr_ast.items():
                for ast_info in ast_infos:
                    start_ast, end_ast = ast_info
                    value = get_type_default_value(type_str)
                    if value is not None:
                        debug_print(start_ast, '->', value)
                        start_ast.update(value)

            debug_print("filter_name_value_ast ************************")
            for attr_name, ast_infos in filter_name_value_ast.items():
                for ast_info in ast_infos:
                    start_ast, end_ast = ast_info
                    value = get_attr_default_value(attr_name)
                    if value is not None:
                        debug_print(start_ast, '->', value)
                        start_ast.update(value)
                    # print(attr_name, start_ast, end_ast)

            debug_print("name_value_map ************************")
            for attr_name, ast_infos in name_value_ast.items():
                for ast_info in ast_infos:
                    start_ast, end_ast = ast_info
                    # print(attr_name, start_ast.get_start_loc(), start_ast, end_ast.get_start_loc())
                    value = get_attr_default_value(attr_name)
                    if value is not None:
                        debug_print(start_ast, '->', value)
                        start_ast.update(value)

        for record_name in self.asts:
            debug_print(self.asts[record_name])

    def run(self, maps):
        self.remove_bits()
        self.set_type_to_default_type()
        self.set_attr_to_default_value(**maps)
        return self.asts

    def set_attr_to_input_value(self, name_value_map, input_inst_attrs):
        can_ignore_attrs = {
            'hasForbiddenSlot', 'FormBits', 'isCTI', 'hasFCCRegOperand', 'Form', 'Itinerary', 'IsPCRelativeLoad',
            'DecoderMethod', 'HardFloatPredicate', 'GPRPredicates', 'EncodingPredicates', 'PTRPredicates',
            'SYMPredicates', 'FGRPredicates', 'InsnPredicates', 'ASEPredicate', 'AdditionalPredicates', 'base',
            'offset', 'BaseOpcode', 'opstr'
        }

        from copy import deepcopy
        input_inst_attrs = deepcopy(input_inst_attrs)
        not_found_attrs_in_input = set()
        for attr_name, ast_infos in name_value_map.items():  # {attr_name: [(start_ast, end_ast)]}
            if attr_name not in input_inst_attrs:
                if attr_name not in can_ignore_attrs:
                    not_found_attrs_in_input.add(attr_name)
                continue
            assert len(ast_infos) == 1, "can only have one start node"
            value_ast, attr_ast = ast_infos[0]
            attr_value = input_inst_attrs[attr_name]

            if attr_value is not None:
                input_inst_attrs["_attrs_set_by_kv"].add(attr_name)
                # print(attr_name, value_ast, '->', attr_value)
                value_ast.update(attr_value)
        # if len(not_found_attrs_in_input) > 0:
        #     print("not found attr in input:", not_found_attrs_in_input)

        for attr_name, attr_value in input_inst_attrs.items():
            if attr_name in attr_default_value_dict:  # ignored name
                if str(attr_value) == attr_default_value_dict[attr_name]:
                    input_inst_attrs[attr_name] = None
        return self.asts, input_inst_attrs
