# 数据清洗
def predicates_transform(data):
    for inst_name, inst_attrs in data.items():
        inst_attrs['Predicates'] = inst_attrs['EncodingPredicates']
        del inst_attrs['EncodingPredicates']


def key_lower_transform(data):
    """
    将部分属性名小写化
    """
    lower_keys = ['rname_upper', 'rname']
    for inst_name, inst_attrs in data.items():
        for key, value in list(inst_attrs.items()):
            if key not in lower_keys and key.lower() in lower_keys:
                inst_attrs[key.lower()] = value
                del inst_attrs[key]


# 格式转换

def out_ins_asmstr_pattern_transform(data):
    from gen.utils import get_tokens, split_by_comma, get_type

    def parser_asm_format(asmstr, instr_attrs):
        """
        解析输入的汇编码形式，提取op,op形式有如下三种
        op | (op) | op1(op2)
        is_r_s_same表示自运算，第一个op既是d又是s
        has_return_value表示指令是否有d op
        is_r_s_same  has_return_value
        0            0                 outs 为空， op 全ins
        0            1                 opts 为op1  其余ins
        1            0                 不存在
        1            1                 outs = op1_alias， op 全ins
        """
        tokens = get_tokens(asmstr)
        is_r_s_same, has_return_value = instr_attrs["Self_Cal"], instr_attrs["Return_Val"]
        op_list, outs_list, ins_list = [], [], []
        asm_str = ""

        if len(tokens) > 1:
            ops_tokens = tokens[1:]
            ops = split_by_comma(ops_tokens)

            for i, op in enumerate(ops):
                if i > 0:
                    asm_str += ", "
                if len(op) == 1:  # rx
                    op_list.append(op[0])
                    asm_str += "$" + op[0]
                elif len(op) == 3:  # (rx)
                    op_list.append(op[1])
                    asm_str += "(${" + op[1] + "})"
                elif len(op) == 4:  # offset(rx)
                    op_list.append(op[2])
                    op_list.append(op[0])
                    asm_str += "${" + op[0] + "}(${" + op[2] + "})"
                else:
                    raise ValueError("op format uncatched!")

            if is_r_s_same == 0:  # 如果没有自操作
                if has_return_value == 1:
                    outs_list.append(op_list[0])
            elif is_r_s_same == 1:  # 有自操作:
                if has_return_value == 1:
                    op_alias = op_list[0] + "_alias"
                    outs_list.append(op_alias)
            idx = len(outs_list)
            if is_r_s_same == has_return_value:
                ins_list.extend(op_list)
            else:
                ins_list.extend(op_list[idx:])

        outs = ", ".join([get_type(op, instr_attrs) + ":$" + op for i, op in enumerate(outs_list)])
        ins = ", ".join([get_type(op, instr_attrs) + ":$" + op for i, op in enumerate(ins_list)])
        asm_str = f'{instr_attrs["rname"]} ' + asm_str

        return f"(outs {outs})", f"(ins {ins})", f'"{asm_str.strip()}"'

    # TODO: add 'Constraint' for 自运算
    for inst_name, inst_attrs in data.items():
        # asmstr = inst_attrs['asmstr']
        # outs, ins, asmstr = parser_asm_format(asmstr, inst_attrs)
        # if 'OutOperandList' not in inst_attrs:
        #     inst_attrs['OutOperandList'] = outs
        # if 'InOperandList' not in inst_attrs:
        #     inst_attrs['InOperandList'] = ins
        # if 'AsmString' not in inst_attrs:
        #     inst_attrs['AsmString'] = asmstr
        inst_attrs['Pattern'] = "[]"
        # print(outs, ins, asmstr)


def TwoOperandAliasConstraint_tansform(data):
    for inst_name, inst_attrs in data.items():
        d = inst_attrs['TwoOperandAliasConstraint']
        if d is not None:
            inst_attrs['TwoOperandAliasConstraint'] = d.replace('?', '')


def format_transform(data):
    for inst_name, inst_attrs in data.items():
        inst_attrs['Format'] = f"InstFormat{inst_attrs['Format']}"


def add_mid_var_transform(data):
    for inst_name, inst_attrs in data.items():
        inst_attrs['_attrs_set_by_kv'] = set()  # used during replace_with_kv and generate body of template


def input_transform(data, data_clean=True):
    if data_clean:
        predicates_transform(data)
        key_lower_transform(data)

    out_ins_asmstr_pattern_transform(data)
    TwoOperandAliasConstraint_tansform(data)
    format_transform(data)
    add_mid_var_transform(data)

