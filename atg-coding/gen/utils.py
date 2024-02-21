
def isCharValue(c: str):
    return c.isalnum() or c == '_' or c == "."


def get_tokens(line):
    """
    获取一行中的所有token，把一个标点符号单算一个token
    """
    tokens = []
    s = ''
    for c in line:
        if isCharValue(c):
            s = s + c
        else:
            if len(s) > 0:
                tokens.append(s)
            if not c.isspace():
                tokens.append(c)
            s = ''
    # 处理最后一个token是标识符
    if len(s) > 0:
        tokens.append(s)
    return tokens


def split_by_comma(token_list):
    # 第一步，按逗号分隔开
    arg_list = []
    arg = []
    for i, token in enumerate(token_list):
        if token == ",":
            arg_list.append(arg)
            arg = []
        else:
            arg.append(token)
    arg_list.append(arg)

    return arg_list


def get_type(name, instr_attrs):
    # print("for debug genasm 66:", feas.name, op)
    op_type = ""
    assert name
    frags = instr_attrs["frags"]
    for frag in frags:
        if name == frag["name"] and frag["is_op"] == True:
            op_type = frag["class_name_riscv"]
            if frag["op_type"] == "Reg":
                op_type = op_type.upper()
    return op_type
