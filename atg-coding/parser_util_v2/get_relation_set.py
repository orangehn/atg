# close_items = {'{': '}', '[': ']', '<': '>', '(': ')'}
# from parser_util.dataset_constructor import *
from parser_util_v2.dataset_constructor import *


# class SuperToken(list):
#     def __init__.py(self, tokens):
#         super(SuperToken, self).__init__.py(tokens)
#         self.du = None
#         self.color = TokenColor.NOCOLOR
#
#
# def collect_super_token(record, close_items, start_id=0, end_item=None):
#     # 递归合并SuperToken
#     i = start_id
#     collected_tokens = []
#
#     while i < len(record):
#         token = record[i]
#         if token == end_item:
#             break
#         elif token in close_items:
#             stokens, i = collect_super_token(record, close_items, i + 1, close_items[token])
#             super_token = SuperToken([token] + stokens + [record[i]])
#             collected_tokens.append(super_token)
#         else:
#             collected_tokens.append(token)
#         i += 1
#     return collected_tokens, i


def set_token(token, du=None, color=None):
    if du is not None:
        token.du = du
    if color is not None:
        token.color = color


def isidentifier(token: str):
    return token.isidentifier()


#
# def set_local(super_token):
#     for token in super_token:
#         if isinstance(token, SuperToken):
#             set_local(token)
#         else:
#             token.local = True


# def set_dataset_loc(dataset):
#     for (filename, records) in dataset:
#         for record_id, record in enumerate(records):
#             for j, token in enumerate(record):
#                 token.loc = Location.create_location(filename, record_id, j)


def func_def_pattern(record, kw_for_def=["def", "class"]):
    """
    function def pattern: KW A<X[,Y]>
    """
    i = 0
    if record[i] in kw_for_def:
        if isidentifier(record[i + 1]):
            KW, A = record[i], record[i + 1]
            set_token(A, 'D', TokenColor.GREEN)
    else:
        LOG.log(LOG.ERROR, '(func_def_pattern)')
        print(record[i].loc)
        raise ValueError("Record starts keyword neither def nor class, but", record[i])


def is_param_scope(record, i):
    """
    <X*>
    """
    if i >= len(record):
        return False
    token = record[i]
    return isinstance(token, SuperToken) and len(token) >= 2 and \
           token[0] == '<' and token[-1] == '>'


def func_use_pattern(record):
    """
    function use pattern: : B<X[, Y]>, C<...>
    """
    # def set_func_use(B, super_token, func_args):
    #     arg_id = 0
    #     for k in range(1, len(super_token) - 1):
    #         if super_token[k] == ',':
    #             arg_id += 1
    #         else:
    #             super_token[k].to = func_args[B][arg_id]

    i = 0
    while i < len(record):
        if record[i] == ':':
            j = i + 1
            while isinstance(record[j], Token) and isidentifier(record[j]):
                B = record[j]
                set_token(B, 'U', TokenColor.RED)
                j += 1
                # if is_param_scope(record, j):
                #     set_func_use(B, record[j])
                if record[j] == ',':
                    j += 1
                else:
                    break
        i += 1


def set_some_du_and_color(dataset):
    for (filename, records) in dataset:
        for record_id, record in enumerate(records):
            record = SuperToken.collect_super_token(record)
            func_def_pattern(record, ["def", "class"])
            func_use_pattern(record)


def get_def_table(dataset):
    """
    将def分成global_def_table:记录间的DU关系
             local_def_table:记录内的DU关系
    """
    global_def_table = {}
    for (filename, records) in dataset:
        for record_id, record in enumerate(records):
            for j, token in enumerate(record):
                if token.local == False and token.du == 'D':
                    global_def_table[token] = token.loc
    return global_def_table


public_class = {'DwarfRegNum', 'DwarfRegAlias', 'DForm_5', 'Register', 'SubtargetFeature', 'Target', 'RegisterOperand', 'Requires',
                'PatLeaf', 'IForm_and_DForm_1', 'AsmParserVariant', 'PointerLikeRegClass', 'XFormMemOp', 'ComplexPattern',
                'Operand', 'InstrMapping', 'InstrInfo', 'AsmParser', 'Instruction', 'AsmOperandClass', 'DwarfRegNum',
                'Predicate', 'RegisterClass', 'SubRegIndex', 'InstrItinClass', 'AssemblerPredicate', 'RegisterWithSubRegs',
                'DwarfRegNum', 'RegisterClass', 'Register', 'SubRegIndex', 'RegisterOperand', 'AsmOperandClass',
                'RegisterWithSubRegs', "ProcessorModel", "InstAlias"}



def get_use_table(dataset, global_def_table):
    # print('R6MMR6Rel' in global_def_table, global_def_table)
    global_use_table = {}
    for (filename, records) in dataset:
        for record_id, record in enumerate(records):
            for j, token in enumerate(record):
                if token.du == 'U':
                    assert isinstance(token, Token)
                    loc = Location.create_location(filename, record_id, j)
                    assert (token in global_def_table) or (token in public_class), f"{token} Use Before Def, Attention!, {record}"
                    if token in global_use_table:
                        global_use_table[token].append(loc)
                    else:
                        global_use_table[token] = [loc]
    return global_use_table


def get_relation_set(dataset, inst_list, global_def_table, i=0):
    relation_set = set()
    # record_set = set()
    #     print(inst_list)
    for inst in inst_list:
        if inst in global_def_table:
            loc = global_def_table[inst]
            print(i, inst, " ".join(dataset.get_record(loc)))
            record = dataset[loc.filename][loc.record_id]
            relation_set = relation_set | {loc}
            tokens = []
            #             print(record)
            for token in record:
                # print(token, token.du)
                if token in global_def_table and token not in public_class:
                    # loc1 = get_nearest_neighbour_loc(token, record_id)
                    loc1 = global_def_table[token]
                    if loc.record_id != loc1.record_id:
                        relation_set = relation_set | {loc}
                        tokens.append(token)
            if tokens:
                # print(relation_set, [dataset.get_tokens_by_loc(loc) for loc in relation_set])
                relation_set = relation_set | (get_relation_set(dataset, tokens, global_def_table, i+1))
    return relation_set


if __name__ == "__main__":
    pass
