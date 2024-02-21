from collections import namedtuple
from collections import OrderedDict

"""
class ADD_: Instruction{
    let xxx;
    let xxx;
}

def ADD: ADD_, xx, 
"""


def get_super_class_for_common(data):
    super_class_map = {
        'RVInst': 'RVInst<(outs), (ins), "", "", [], InstFormatR>',
        'RVInstR': 'RVInstR<0, 0, OPC_LOAD, (outs), (ins), "", "">',
        'RVInstR4': 'RVInstR4<0, OPC_LOAD, (outs), (ins), "", "">',
        'RVInstRAtomic': 'RVInstRAtomic<0, 0, 0, 0, OPC_LOAD, (outs), (ins), "", "">',
        'RVInstRFrm': 'RVInstRFrm<0, OPC_LOAD, (outs), (ins), "", "">',
        'RVInstI': 'RVInstI<0, OPC_LOAD, (outs), (ins), "", "">',
        'RVInstIShift': 'RVInstIShift<0, 0, OPC_LOAD, (outs), (ins), "", "">',
        'RVInstIShiftW': 'RVInstIShiftW<0, 0, OPC_LOAD, (outs), (ins), "", "">',
        'RVInstS': 'RVInstS<0, OPC_LOAD, (outs), (ins), "", "">',
        'RVInstB': 'RVInstB<0, OPC_LOAD, (outs), (ins), "", "">',
        'RVInstU': 'RVInstU<OPC_LOAD, (outs), (ins), "", "">',
        'RVInstJ': 'RVInstJ<OPC_LOAD, (outs), (ins), "", "">',

        'RVInst16': 'RVInst16<(outs), (ins), "", "", [], InstFormatCR>',

        'RVInstVV': 'RVInstVV<0, OPIVV, (outs), (ins), "", "">',
        "RVInstIVI": 'RVInstIVI<0, (outs), (ins), "", "">',
    }
    if 'rsuper' in data:
        if data['rsuper'] in super_class_map:
            super_class = super_class_map[data['rsuper']]
        elif data['rsuper'].startswith('RVInst16'):
            super_class = super_class_map['RVInst16']
        elif data['rsuper'].startswith('RVInst'):
            super_class = super_class_map['RVInst']
        else:
            raise ValueError(f"{data['rsuper']} is not found.")
    else:
        super_class = 'Instruction'
    return super_class


class CommonTemplate(object):
    def __init__(self, args=[
    "Size", "Namespace", "OutOperandList", "InOperandList", "AsmString", "Pattern",
    "Predicates",
    "hasSideEffects", "mayLoad",
    "mayStore", "isCall", "hasDelaySlot",
    "isReMaterializable", "isAsCheapAsAMove", "isBarrier",
    "isReturn", "isTerminator", "isBranch", "isIndirectBranch",
    "Defs", "Uses", "SoftFail", "isCommutable", "AddedComplexity",
    "isEHScopeReturn", "isCompare", "isMoveImm", "isMoveReg",
    "isBitcast", "isSelect", "isAdd", "isTrap", "canFoldAsLoad",
    "mayRaiseFPException", "isConvertibleToThreeAddress",
    "isPredicable", "isUnpredicable", "usesCustomInserter",
    "hasPostISelHook", "hasCtrlDep", "isNotDuplicable",
    "isConvergent", "hasExtraSrcRegAllocReq", "hasExtraDefRegAllocReq",
    "isRegSequence", "isPseudo", "isExtractSubreg", "isInsertSubreg",
    "variadicOpsAreDefs", "isCodeGenOnly", "isAsmParserOnly",
    "hasNoSchedulingInfo", "UseNamedOperandTable", "FastISelShouldIgnore",
    "Arch", "DecoderNamespace", "TwoOperandAliasConstraint", "Constraints"], additional_args=[]):
        self.args = args + additional_args

    def gen_class(self, data):
        stmts = []
        for key in self.args:
            if data[key] is not None:
                stmts.append(f"\tlet {key} = {data[key]};")
        body = "\n".join(stmts)

        shedu = f", Sched<{data['Sched']}>" if data['Sched'] is not None else ""

        class_name = f"{data['rname_upper']}_Inst"
        code = f"class {class_name}: {get_super_class_for_common(data)}{shedu}" + "{\n" + body + "\n}"
        return code

    def gen_call(self, data):
        return f"{data['rname_upper']}_Inst"


class MyInstTemplate(object):
    def __init__(self, name='ATGInst'):
        self.name = name

    def gen_class(self, super_class):
        super_class = f": {super_class}" if len(super_class) > 0 else ""
        return f"class {self.name}{super_class};"

    def gen_call(self, data):
        return f"{self.name}"


class FMTemplate(object):
    def __init__(self):
        pass

    def _get_index_str_of_part(self, part, frag_len):
        if 'range' not in part:
            return ""
        part_len = part["range"][0] - part["range"][1]
        if part_len == frag_len and part["range"][1] == 0:
            suffix = ""
        else:
            suffix = f"{{{part_len}-0}}"
        return suffix

    def gen_class(self, data, super_class):
        def_stmts = [f"bits<{int(data['Size']) * 8}> SoftFail=0;", f"bits<{int(data['Size'])*8}> Inst;"]
        assign_stmts = []
        for frag in data['frags']:
            name, value = frag['name'], frag['value']
            if len(frag['parts']) == 1:  # single part: Inst{a-b} = rd; / Inst{a-b}=0b0101;
                part = frag['parts'][0]

                if name is not None:
                    # length = part["inst_range"][0] - part["inst_range"][1] + 1  # {'inst_range': (19, 15)}
                    assert 'len' in frag, frag
                    length = frag['len']
                    def_stmt = f"bits<{length}> {name}"
                    def_stmt += f" = {value};" if value is not None else ";"
                    def_stmts.append(def_stmt)
                    index = self._get_index_str_of_part(part, length)
                    assign_stmts.append(f'let Inst{{{part["inst_range"][0]}-{part["inst_range"][1]}}} = {name}{index};')
                else:
                    assert value is not None
                    assign_stmts.append(f'let Inst{{{part["inst_range"][0]}-{part["inst_range"][1]}}} = {value};')
            else:  # multi parts:
                """
                name, value, part1(a1, b1, c1, d1), part2(a2, b2, c2, d2):
                way1:
                    Inst{a1-b1} = value{c1-d1};
                    Inst{a2-b2} = value{c2-d2};
                way2:
                    bits<len> name = value;
                    Inst{a1-b1} = name{c1-d1};
                    Inst{a2-b2} = name{c2-d2};
                """
                # way = 1
                # if way == 1:  # way1, only for name is None
                #     print(frag['parts'])
                #     for part in frag['parts']:
                #         right_high_value = value[2:][::-1]
                #         part_e, part_s = part["range"]
                #         assign_stmts.append(f'let Inst{{{part["inst_range"][0]}-{part["inst_range"][1]}}}'
                #                             f' = 0b{right_high_value[part_s:part_e+1][::-1]};')
                # elif way == 2:  # way2
                assert name is not None
                # length = 0
                # for part in frag['parts']:
                #     length += part["inst_range"][0] - part["inst_range"][1] + 1  # {'inst_range': (19, 15)}
                assert 'len' in frag, frag
                length = frag['len']
                stmt = f"bits<{length}> {name}"
                stmt += ';' if value is None else f' = {value};'
                def_stmts.append(stmt)

                for part in frag['parts']:
                    assign_stmts.append(f'let Inst{{{part["inst_range"][0]}-{part["inst_range"][1]}}}'
                                        f' = {name}{{{part["range"][0]}-{part["range"][1]}}};')
                # else:
                    # raise ValueError

        body = "\n".join(['\t' + stmt for stmt in def_stmts + assign_stmts])
        class_name = f"{data['rname_upper']}_FM2"
        super_class = f': {super_class}' if len(super_class) > 0 else ''
        code = f"class {class_name}{super_class}{{\n{body}\n}}"
        return code

    def gen_call(self, data):
        return f"{data['rname_upper']}_FM2"


class Arg(object):
    def __init__(self, type, arg_name, default_value=None):
        self.type = type
        self.arg_name = arg_name
        self.default_value = default_value


class Template(object):
    def __init__(self, name, args_dict: OrderedDict):
        self.name = name
        self.args_dict = args_dict

    def gen_def_arg_list(self):
        args = []
        for attr_name, v in self.args_dict.items():
            arg = f'{v.type} {v.arg_name}'
            if v.default_value is not None:
                arg += f' = {v.default_value}'
            args.append(arg)
        return ', '.join(args)

    def gen_call_arg_list(self, data):
        args = []
        last_args_is_same_as_default = True
        for attr_name, v in list(self.args_dict.items())[::-1]:
            # 1. last args is same as default 2. current arg have default value
            if last_args_is_same_as_default and v.default_value is not None:
                if attr_name not in data:  # if not have such arg, use default value, continue ignore.
                    continue
                else:  # if have such arg, current arg is same as default value, continue ignore.
                    if data[attr_name] == v.default_value:
                        continue
            last_args_is_same_as_default = False
            args.append(data[attr_name])
        return ', '.join(args[::-1])


class TFFlagTemplate(Template):
    def __init__(self, name='TSFlagTemplate', args_dict=OrderedDict(
        Format=Arg('InstFormat', 'format'),
        RVVConstraint=Arg('RISCVVConstraint', 'rvv_constraint', 'NoConstraint')
    )):
        super(TFFlagTemplate, self).__init__(name, args_dict)

    def gen_class(self):
        code = f"class {self.name}<{self.gen_def_arg_list()}>" + \
              """{
  bits<64> TSFlags = 0;
  let TSFlags{4-0} = format.Value;

  // Defaults
  RISCVVConstraint RVVConstraint = rvv_constraint;
  let TSFlags{7-5} = RVVConstraint.Value;

  bits<3> VLMul = 0;
  let TSFlags{10-8} = VLMul;

  bit HasDummyMask = 0;
  let TSFlags{11} = HasDummyMask;

  bit WritesElement0 = 0;
  let TSFlags{12} = WritesElement0;

  bit HasMergeOp = 0;
  let TSFlags{13} = HasMergeOp;

  bit HasSEWOp = 0;
  let TSFlags{14} = HasSEWOp;

  bit HasVLOp = 0;
  let TSFlags{15} = HasVLOp;
}
                  """
        return code

    def gen_call(self, data):
        return f"{self.name}<{self.gen_call_arg_list(data)}>"


class DependenceTemplate(Template):
    def __init__(self, name='DepTemplate', args_dict=OrderedDict()):
        super(DependenceTemplate, self).__init__(name, args_dict)

    def gen_class(self):
        code = """
def addr :
  ComplexPattern<iPTR, 2, "selectIntAddr", [frameindex]>;

def addrRegImm :
  ComplexPattern<iPTR, 2, "selectAddrRegImm", [frameindex]>;

def addrDefault :
  ComplexPattern<iPTR, 2, "selectAddrDefault", [frameindex]>;

def addrimm10 : ComplexPattern<iPTR, 2, "selectIntAddrSImm10", [frameindex]>;
def addrimm10lsl1 : ComplexPattern<iPTR, 2, "selectIntAddrSImm10Lsl1",
                                   [frameindex]>;
def addrimm10lsl2 : ComplexPattern<iPTR, 2, "selectIntAddrSImm10Lsl2",
                                   [frameindex]>;
def addrimm10lsl3 : ComplexPattern<iPTR, 2, "selectIntAddrSImm10Lsl3",
                                   [frameindex]>;
                  """
        return code


class TypeFormatTemplate(Template):
    def __init__(self, name="Type_AUX_FM", args_dict=OrderedDict()):
        super(TypeFormatTemplate, self).__init__(name, args_dict)

    def gen_class(self):
        code = f"class {self.name}" + """{
    bit Value = 0;
}
        """
        return code

    def gen_call(self, data):
        return f"{self.name}<{self.gen_call_arg_list(data)}>"


class EndTemplate(object):
    """
    args
    [
        'DecoderNamespace', 'Itinerary', 'Namespace', 'isCommutable', 'Size', 'OutOperandList', 'InOperandList',
        'AsmString', 'Pattern', 'isReMaterializable', 'TwoOperandAliasConstraint', 'Constraints', 'isAsCheapAsAMove',
        'Predicates', 'Defs', 'isBranch', 'isTerminator', 'isBarrier', 'hasDelaySlot', 'isCall', 'DecoderMethod',
        'hasSideEffects', 'isMoveReg', 'isCompare', 'usesCustomInserter', 'isCodeGenOnly', 'hasPostISelHook',
        'isIndirectBranch', 'canFoldAsLoad', 'mayLoad', 'AddedComplexity', 'hasNoSchedulingInfo', 'mayStore',
        'AsmMatchConverter', 'Uses', 'DisableEncoding']
    """
    LEFT_ATTR = set()
    def __init__(self, args=[
    "Size", "Namespace", "OutOperandList", "InOperandList", "AsmString", "Pattern",
    "Predicates",
    "hasSideEffects", "mayLoad",
    "mayStore", "isCall", "hasDelaySlot",
    "isReMaterializable", "isAsCheapAsAMove", "isBarrier",
    "isReturn", "isTerminator", "isBranch", "isIndirectBranch",
    "Defs", "Uses", "SoftFail", "isCommutable", "AddedComplexity",
    "isEHScopeReturn", "isCompare", "isMoveImm", "isMoveReg",
    "isBitcast", "isSelect", "isAdd", "isTrap", "canFoldAsLoad",
    "mayRaiseFPException", "isConvertibleToThreeAddress",
    "isPredicable", "isUnpredicable", "usesCustomInserter",
    "hasPostISelHook", "hasCtrlDep", "isNotDuplicable",
    "isConvergent", "hasExtraSrcRegAllocReq", "hasExtraDefRegAllocReq",
    "isRegSequence", "isPseudo", "isExtractSubreg", "isInsertSubreg",
    "variadicOpsAreDefs", "isCodeGenOnly", "isAsmParserOnly",
    "hasNoSchedulingInfo", "UseNamedOperandTable", "FastISelShouldIgnore",
    "Arch", "DecoderNamespace", "TwoOperandAliasConstraint", "Constraints"], additional_args=[]):
        self.args = args + additional_args

    def get_aux_class_name(self, data):
        class_name = f"{data['rname_upper']}"
        return f"{class_name}_AUX"

    def gen_code(self, data):
        stmts = []
        left_args = set(self.args) - data["_attrs_set_by_kv"]
        for key in data:  # if attr not set by kv replace
            if data[key] is not None and key in left_args:
                stmts.append(f"\tlet {key} = {data[key]};")
                EndTemplate.LEFT_ATTR.add(key)
                # print(f'add {key}')
        body = "\n".join(stmts)
        # print(EndTemplate.LEFT_ATTR)
        return body

    def gen_def(self, data, call_list, let_attrs=[]):
        assert len(let_attrs) == 0
        body = self.gen_code(data)
        class_name = f"{data['rname_upper']}"
        shedu = f", Sched<{data['Sched']}>" if data['Sched'] is not None else ""
        code = f"def {class_name}: {', '.join(call_list)}{shedu}" + "{\n" + body + "\n}"
        return code

    def gen_aux_class(self, data, call_list, let_attrs=[]):
        assert len(let_attrs) == 0
        body = self.gen_code(data)
        shedu = f", Sched<{data['Sched']}>" if data['Sched'] is not None else ""
        def_stmt = f"class {self.get_aux_class_name(data)}: {', '.join(call_list)}{shedu};"
        return def_stmt, body

    def gen_call(self, data, call_list):
        code = self.gen_code(data)
        code = f"{', '.join(call_list)}{code}"
        return code
