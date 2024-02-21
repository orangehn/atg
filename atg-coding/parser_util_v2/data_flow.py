from .get_relation_set import *

keywords = ["class", "def", "let", "defm", "multiclass", "foreach"]


def is_func_form(token1, token2):
    """
    token1 identifier, token2 supertoken<>
    """
    return isidentifier(token1) and isinstance(token2, SuperToken) and token2[0] == '<' and token2[-1] == '>'


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


def func_decl_arg_loc(record):
    """
    KW A<X,Y>
    """

    def arg_parse(token_list):
        """
        string opcode, bits<5> funct3
        """
        #         print(token_list)
        arg_list = split_by_comma(token_list)
        #         print(arg_list)
        # 去掉默认值
        arg_new_list = []
        for arg in arg_list:
            for i, token in enumerate(arg):
                if token == "=":
                    arg = arg[0:i]
                    break
            arg_new_list.append(arg)
        return [arg[-1].loc for arg in arg_new_list]

    decl_arg_map = {}
    record_func_map = {record[0].loc: []}
    for i, token in enumerate(record):
        if token in keywords:
            if is_func_form(record[i + 1], record[i + 2]):
                record_func_map[record[0].loc].append(record[i + 1])  # 此处获得record位置与定义的func name的映射，
                arg_loc_list = arg_parse(record[i + 2][1:-1])
                decl_arg_map[record[i + 1]] = arg_loc_list

    return decl_arg_map, record_func_map


def get_first_token(token):
    token = token[0]
    while isinstance(token, SuperToken):
        token = token[0]
    return token


def get_length(token_list):
    """
    获得包含supertoken的tokens序列长度
    """
    length = 0
    for token in token_list:
        if isinstance(token, SuperToken):
            l = get_length(token)
        else:
            l = 1
        length += l
    return length


def func_use_arg_loc(record):
    """
    :B<X,Y>, C...
    """

    def arg_parse(token_list):
        """
        opcode, (), [], 1
        """
        # 第一步按逗号分隔开
        arg_loc_list = []
        arg_list = split_by_comma(token_list)
        for arg in arg_list:  # 其实arg里就对应一个参数
            if len(arg) == 1:
                if isinstance(arg[0], SuperToken):  # (outs RO:$rd),
                    length = get_length(arg[0])
                    token = get_first_token(arg)
                    loc = token.loc
                    loc = Location.create_location(loc.filename, loc.record_id, loc.token_id, length)
                    arg_loc_list.append(loc)
                else:  # opcode
                    arg_loc_list.append(arg[0].loc)
            elif len(arg) > 1:  # !strconcat(opstr, "\t$rd, $rs, $rt"),
                length = get_length(arg[0])
                token = get_first_token(arg)
                loc = token.loc
                loc = Location.create_location(loc.filename, loc.record_id, loc.token_id, length)
                arg_loc_list.append(loc)
            else:
                raise ValueError("length invalid!")
        return arg_loc_list

    use_arg_dict = {}
    for i, token in enumerate(record):
        if token == ":":
            j = i + 1
            if is_func_form(record[j], record[j + 1]):
                arg_loc_list = arg_parse(record[j + 1][1:-1])
                use_arg_dict[record[j]] = arg_loc_list
                j = j + 2
            else:
                j = j + 1
            while record[j] == ",":
                j = j + 1
                if is_func_form(record[j], record[j + 1]):
                    arg_loc_list = arg_parse(record[j + 1][1:-1])
                    use_arg_dict[record[j]] = arg_loc_list
                    j = j + 2
                else:
                    j = j + 1

    return use_arg_dict


def func_use_decl_flow(dataset):
    """
    func_use - > func_decl
    """
    decl_arg_map = {}
    use_arg_map = {}
    record_func_map = {}
    for filename, records in dataset:
        for record in records:
            record = SuperToken.collect_super_token(record)
            decl_arg_dict, record_func_dict = func_decl_arg_loc(record)
            decl_arg_map.update(decl_arg_dict)
            record_func_map.update(record_func_dict)
            use_arg_dict = func_use_arg_loc(record)
            for name, use_arg_list in use_arg_dict.items():
                if name not in use_arg_map:
                    use_arg_map[name] = [use_arg_list]
                else:
                    use_arg_map[name].append(use_arg_list)

    func_arg_flow_dict = {}
    for name, use_loc_lists in use_arg_map.items():
        if name not in decl_arg_map:
            LOG.log(LOG.WARN, "(func_use_decl_flow)", f"{name} not in decl_arg_map, it may not be defined.")
            continue
        for use_arg_list in use_loc_lists:
            decl_arg_list = decl_arg_map[name]
            for i, use_arg_loc in enumerate(use_arg_list):  # 对每个use_arg的to赋予decl_arg的loc
                func_arg_flow_dict[use_arg_loc] = decl_arg_list[i]  #:B<""> -> B<string opstr>
    return func_arg_flow_dict, decl_arg_map, use_arg_map, record_func_map


def get_token_by_loc(loc, dataset):
    """
    通过location 获得token str
    """
    return dataset[loc.filename][loc.record_id][loc.token_id]


def record_inside_flow(dataset, decl_arg_map, record_func_map):
    """
    class A<string opstr, bit iscom = 0>: B<opstr, (outs $opstr)>{
        let iscommon = iscom;
    }
    """
    record_inside_flow_map = defaultdict(list)
    for filename, records in dataset:
        for record in records:
            # 第一步获得child_arg
            names = record_func_map[record[0].loc]
            if len(names) == 0:
                continue
            child_arg_loc_list = decl_arg_map[names[0]]  # TODO: 没有处理let a = [] in {def ; def; def; ...}
            for arg in child_arg_loc_list:
                stage = 0
                arg_token = dataset[arg.filename][arg.record_id][arg.token_id]
                for i, token in enumerate(record):
                    if token == ":":
                        stage = 1
                    elif token == '{':
                        stage = 2
                    if stage == 1:
                        if token == arg_token:
                            record_inside_flow_map[arg].append(token.loc)
                    elif stage == 2:
                        if token == '=':
                            stage = 3
                    elif stage == 3:
                        if token == arg_token:
                            record_inside_flow_map[arg].append(token.loc)
                        stage = 2

    return record_inside_flow_map


# def one_record_inside_flow(record, dataset, decl_arg_map, record_func_map):
#     """
#     class A<string opstr, bit iscom = 0>: B<opstr, (outs $opstr)>{
#         let iscommon = iscom;
#     }
#     """
#     record_inside_flow_map = {}
#
#     # 第一步获得child_arg
#     names = record_func_map[record[0].loc]
#
#     child_arg_loc_list = decl_arg_map[names[0]]  # TODO: 没有处理let a = [] in {def ; def; def; ...}
#
#     for arg in child_arg_loc_list:
#         stage = 0
#         arg_token = dataset[arg.filename][arg.record_id][arg.token_id]
#         for i, token in enumerate(record):
#             if token == ":":
#                 stage = 1
#             if stage == 1:
#                 if token == "opstr":
#                     print("opstr", token.loc)
#                 if token == arg_token:
#                     record_inside_flow_map[arg] = token.loc
#
#     return record_inside_flow_map


def isSymbol(token: str):
    return len(token) == 1 and not (token.isalnum() or token == '_')


def find_right_close_token(record, j: int, token):
    """
    [, ,]
    """
    while record[j] != token:
        j = j + 1
        if j > len(record) - 1:  # 判断是否越界
            return None
    return j


def find_left_close_token(record, j, token):
    """
    inst{4-0}
    """
    while record[j] != token:
        j = j - 1
        if j < 0:  # 判断是否越界
            return None
    return j


def record_inside_equal_flow(dataset, decl_arg_map, record_func_map):
    """
    A=B   A=""   inst{4-0}=funct3       Predicate = [NotInMips16Mode]
    """

    def func_contents_range(record):
        """
        parser contents
        input: 未收集supertoken的record
        output: record[start_content:end_content_loc]
        """
        contents_range = []
        sentence_dict = {}
        has_contents = False

        record1 = SuperToken.collect_super_token(record)
        func_name = record1[1]
        for i, token in enumerate(record1):
            if isinstance(token, SuperToken):
                if token[0] == "{":
                    has_contents = True
                    start_loc = token[1].loc.token_id
                    end_loc = token[-1].loc.token_id
                    contents_range.append(start_loc)
                    contents_range.append(end_loc)
                    break
        if has_contents:  # 有函数体
            tokens = record[contents_range[0]:contents_range[1]]
            return tokens

        return None

    close_items = {"[": "]", "(": ")"}
    record_inside_equal_map = {}
    for filename, records in dataset:
        for record in records:
            # record, _ = collect_super_token(record, close_items)
            #             print(record)
            if not func_contents_range(record):
                continue
            record = func_contents_range(record)
            for i, token in enumerate(record):
                if token == "=":
                    #                     print(record[i-1], token, record[i+1])
                    if not isSymbol(record[i - 1]):  # 排除supertoken情况
                        left_loc = record[i - 1].loc
                    elif record[i - 1] == "}":  # inst{4-0}=     TSFlags{3-0}=
                        j = find_left_close_token(record, i - 2, "{")
                        loc = record[j - 1].loc  # inst.loc
                        left_loc = Location.create_location(loc.filename, loc.record_id, loc.token_id, i - j + 1)
                    else:
                        raise TypeError("Type invalid!")

                    if not isSymbol(record[i + 1]) or record[i + 1] == '?':  # # A="BCD" / A=""
                        right_loc = record[i + 1].loc
                    elif record[i + 1] in close_items:
                        j = find_right_close_token(record, i + 2, close_items[record[i + 1]])
                        length = j - i
                        loc = record[i + 1].loc
                        right_loc = Location.create_location(loc.filename, loc.record_id, loc.token_id, length)
                    elif record[i + 1] == '!':  # tablegen函数调用语法
                        j = find_right_close_token(record, i + 2, ';')
                        loc = record[i + 1].loc
                        right_loc = Location.create_location(loc.filename, loc.record_id, loc.token_id, j - i)
                    else:
                        raise TypeError("invalid!")

                    record_inside_equal_map[right_loc] = left_loc  # B -> A
    #             print()
    return record_inside_equal_map


def split_by_semicolon(token_list):
    # 第一步，按分号分隔开
    arg_list = []
    arg = []
    for i, token in enumerate(token_list):
        arg.append(token)
        if token == ";":
            arg_list.append(arg)
            arg = []

    return arg_list


def func_contents_parser(record):
    """
    parser contents
    input: 未收集supertoken的record
    output: contents_range, def_arg_list, let_arg_list
    """
    close_items = {"[": "]", "(": ")"}
    contents_range = []
    has_contents = False

    record1 = SuperToken.collect_super_token(record)
    for i, token in enumerate(record1):
        if isinstance(token, SuperToken):
            if token[0] == "{":
                has_contents = True
                start_loc = token[1].loc.token_id
                end_loc = token[-1].loc.token_id
                contents_range.append(start_loc)
                contents_range.append(end_loc)
                break
    if has_contents:  # 有函数体
        tokens = record[contents_range[0]:contents_range[1]]
        sentence_list = split_by_semicolon(tokens)

        def_arg_map = {}
        right_value_map = {}
        for sentence in sentence_list:
            is_let_start = 1 if sentence[0] == "let" else 0
            find_equal = 0
            for i, token in enumerate(sentence):
                if token == "=":
                    find_equal = 1
                    left_loc = sentence[i - 1].loc
                    if not is_let_start:
                        def_arg_map[left_loc] = sentence[i - 1]
                    # 然后分析等号右值情况，分为标识符、初始值(0/?/"")、或者引号字符(引号视作一个字符)等正常情况，
                    # 或者[] 以及 !打头的函数调用等特殊情况，这种情况都记录下结束位置，在创建location时加上长度
                    if not isSymbol(sentence[i + 1]) or sentence[i + 1] == '?':  # # A="BCD" / A=""
                        right_loc = sentence[i + 1].loc
                        right_token = sentence[i + 1]
                    elif sentence[i + 1] in ["[", "("]:
                        j = find_right_close_token(sentence, i + 2, close_items[sentence[i + 1]])
                        loc = sentence[i + 1].loc
                        right_loc = Location.create_location(loc.filename, loc.record_id, loc.token_id, j - i)
                        right_token = sentence[i + 1:j + 1]
                    elif sentence[i + 1] == '!':  # tablegen函数调用语法
                        j = find_right_close_token(sentence, i + 2, ';')
                        #                         print("******j", type(j), j, fmt_print(sentence))
                        loc = sentence[i + 1].loc
                        right_loc = Location.create_location(loc.filename, loc.record_id, loc.token_id, j - i)
                        right_token = sentence[i + 1:j]
                    else:
                        raise ValueError(sentence[i + 1], sentence)
                    right_value_map[right_loc] = right_token
                    break
            if not find_equal:
                left_loc = sentence[len(sentence) - 2].loc
                def_arg_map[left_loc] = sentence[len(sentence) - 2]

        return def_arg_map, right_value_map

    return None, None


def func_local_flow(dataset):
    """
    函数内部def -> use 局部数据流分析
    bit opstr;
    let opcode = opstr;
    opstr -> opstr
    """
    record_loc_flow_map = {}
    for filename, records in dataset:
        for record in records:
            def_arg_map, right_value_map = func_contents_parser(record)
            if def_arg_map is not None and right_value_map is not None:
                for def_arg_loc, def_arg in def_arg_map.items():
                    for right_value_loc, right_value in right_value_map.items():
                        if def_arg in right_value:
                            record_loc_flow_map[def_arg_loc] = right_value_loc

    return record_loc_flow_map


# bits flow analyse, special because it's based of relation set record
def func_decl_arg_parser2(record):
    """
    KW A<X,Y>
    """

    def decl_arg_parse(token_list):
        """
        string opcode, bits<5> funct3
        """
        #         print(token_list)
        arg_list = split_by_comma(token_list)
        #         print(arg_list)
        # 去掉默认值
        arg_new_list = []
        for arg in arg_list:
            for i, token in enumerate(arg):
                if token == "=":
                    arg = arg[0:i]
                    break
            arg_new_list.append(arg)
        return arg_new_list

    decl_arg_map = {}
    record = SuperToken.collect_super_token(record)
    for i, token in enumerate(record):
        if token in keywords:
            if is_func_form(record[i + 1], record[i + 2]):
                arg_list = decl_arg_parse(record[i + 2][1:-1])
                # print("arg_list********************************")
                # print(arg_list)
                decl_arg_map[record[i + 1]] = arg_list

    return decl_arg_map


def func_contents_parser2(record):
    """
    parser contents
    input: 未收集supertoken的record
    output: contents_range, def_arg_list, let_arg_list
    """
    contents_range = []
    sentence_dict = {}
    has_contents = False

    record1 = SuperToken.collect_super_token(record)
    func_name = record1[1]
    for i, token in enumerate(record1):
        if isinstance(token, SuperToken):
            if token[0] == "{":
                has_contents = True
                start_loc = token[1].loc.token_id
                end_loc = token[-1].loc.token_id
                contents_range.append(start_loc)
                contents_range.append(end_loc)
                break
    if has_contents:  # 有函数体
        tokens = record[contents_range[0]:contents_range[1]]
        sentence_list = split_by_semicolon(tokens)
        sentence_dict[func_name] = sentence_list
        return sentence_dict

    return None


def is_bits_arg_form(token1, token2):
    """
    bits<4> op
    """
    return token1 == "bits" and isinstance(token2, SuperToken) and token2[0] == "<" and token2[-1] == ">"


def is_bits_brace_form(token1, token2):
    """
    Inst [ {31-26} ]
    """
    return isidentifier(token1) and isinstance(token2, SuperToken) and token2[0] == "{" and token2[-1] == "}"


def parser_bits(tokens):
    """
    {4-0} /   {3}
    传过来的参数，不包含花括号
    """
    bit_list = []
    bit_value = tokens[0].loc
    bit_list.append(tokens[0].loc)
    if len(tokens) == 3 and tokens[1] == "-":
        bit_list.append(tokens[2].loc)
    return bit_list


# dataset used for debug
def parser_bits_pattern(records_list, dataset):
    """
    解析bits<XX> Identifier 模式匹配，XX表示位数
    这里需要建立从Identifier -> XX的数据流向，只解决名字问题，因此不需要跟之前的五种数据流结合到一起
    """

    def parser_arg_bits(arg_map):
        """
        解析参数里的bits<>
        """
        arg_bits_dict = {}
        for func_name, arg_tokens_list in arg_map.items():
            for arg_tokens in arg_tokens_list:
                if isinstance(arg_tokens[1], SuperToken):  # bits [ <4> ] op
                    if is_bits_arg_form(arg_tokens[0], arg_tokens[1]):
                        name_loc = arg_tokens[2].loc
                        name_str = dataset[name_loc.filename][name_loc.record_id][name_loc.token_id]
                        loc = arg_tokens[1][1].loc
                        arg_bits_dict[name_str] = [loc]
                else:
                    i = 0
                    if arg_tokens[0] == "field":  # field bits<32> inst
                        i = 1
                    if arg_tokens[i] == "bits" and arg_tokens[i + 1] == "<" and arg_tokens[
                        i + 3] == ">":  # bits<32> inst
                        name_loc = arg_tokens[i + 4].loc
                        name_str = dataset[name_loc.filename][name_loc.record_id][name_loc.token_id]
                        loc = arg_tokens[i + 2].loc
                        arg_bits_dict[name_str] = [loc]
        return arg_bits_dict

    def parser_contents_bits(content_dict):
        """
        let inst{31-26} = opcode
        """
        contents_bits_dict = {}
        for func_name, contents_list in content_dict.items():
            for content in contents_list:
                if content[0] == "let":
                    if not isinstance(content[2], SuperToken):
                        content = SuperToken.collect_super_token(content)
                    if is_bits_brace_form(content[1], content[2]):
                        bit_list = parser_bits(content[2][1:-1])
                        name1_loc = content[1].loc
                        name2_loc = content[4].loc
                        # name_loc = (name1_loc, name2_loc)
                        name1 = dataset[name1_loc.filename][name1_loc.record_id][name1_loc.token_id]
                        name2 = dataset[name2_loc.filename][name2_loc.record_id][name2_loc.token_id]
                        name_str = name1 + "-" + name2
                        contents_bits_dict[name_str] = bit_list
        return contents_bits_dict

    bits_dict = {}  # funct_name: loc /loc_list
    for record in records_list:
        record.print()
        decl_arg_map = func_decl_arg_parser2(record)  # 获得形参列表 func_name: arg_list[]
        sentence_dict = func_contents_parser2(record)  # 获得函数体语句，func_name : list[]
        if decl_arg_map:
            arg_bits_dict = parser_arg_bits(decl_arg_map)  # 获得bits<5> = op  中 op:5  name:loc的映射
            bits_dict.update(arg_bits_dict)

        if sentence_dict:
            content_arg_bits_dict = parser_arg_bits(sentence_dict)
            content_bits_dict = parser_contents_bits(sentence_dict)
            bits_dict.update(content_arg_bits_dict)
            bits_dict.update(content_bits_dict)

    return {k: [tuple(v)] for k, v in bits_dict.items()}


# end bits flow analyse


# def parser_let_flow(dataset):
#     """
#      let a = 1 in def A;
#      put 1 -> a into data flow of A
#     """
#
#     def let_split_by_comma(token_list):
#         # 第一步，按逗号分隔开
#         arg_list = []
#         arg = []
#         token_list, _ = SuperToken.collect_super_token(token_list)
#         for i, token in enumerate(token_list):
#             if token == ",":
#                 arg_list.append(arg)
#                 arg = []
#             else:
#                 arg.append(token)
#         arg_list.append(arg)
#
#         return arg_list
#
#     close_items = {"[": "]", "(": ")"}
#     let_constraint_flow_map = {}
#     for filename, records in dataset:
#         for record in records:
#             if record.data:
#                 let_constraint = record.data["let"][0]
#                 let_constraint = SuperToken.unfold_super_token(let_constraint)
#                 print(let_constraint)
#                 i = 0
#                 while i < len(let_constraint):
#                     token = let_constraint[i]
#                     if token == "in":
#                         break
#                     if token == "=":
#                         if not isSymbol(let_constraint[i - 1]):  # 排除supertoken情况
#                             left_loc = let_constraint[i - 1].loc
#                         else:
#                             raise TypeError("Let left Type invalid!")
#
#                         if not isSymbol(let_constraint[i + 1]):  # # A="BCD" / A=1
#                             right_loc = let_constraint[i + 1].loc
#                             i = i + 2
#                         elif let_constraint[i + 1] in close_items:
#                             j = find_right_close_token(let_constraint, i + 2, close_items[let_constraint[i + 1]])
#                             length = j - i
#                             loc = let_constraint[i + 1].loc
#                             print(let_constraint[i + 1], type(let_constraint[i + 1]), let_constraint[i + 1].loc)
#                             right_loc = Location.create_location(loc.filename, loc.record_id, loc.token_id, length)
#                             i = j + 1
#                         # elif record[i + 1] == '!':  # tablegen函数调用语法
#                         #     j = find_right_close_token(record, i + 2, ';')
#                         #     loc = record[i + 1].loc
#                         #     right_loc = Location.create_location(loc.filename, loc.record_id, loc.token_id, j - i)
#                         else:
#                             raise TypeError("invalid!")
#
#                         let_constraint_flow_map[right_loc] = left_loc
#                     else:
#                         i += 1
#             else:
#                 continue
#
#     return let_constraint_flow_map


# get data flow from data_flow
def get_rt_flow(relation_records, data_flow, global_def_table, dataset):
    """
    按照record_id，把数据流从data_flow种找出
    """
    rt_data_flow = {}
    for from_loc, to_loc in data_flow.items():
        for record in relation_records:
            record_loc = record[0].loc
            if record_loc.record_id == from_loc.record_id and record_loc.filename == from_loc.filename:
                token = dataset[from_loc.filename][from_loc.record_id][from_loc.token_id]
                if token in global_def_table:
                    continue
                rt_data_flow[from_loc] = to_loc
                break

    return rt_data_flow


def get_flow_chain(data_flow):
    """
    f  t
    f  t
    f  t
    """
    def deep_traverse(tree, start):
        if start not in tree: return [[start]]
        locs = tree[start]
        all_chains = []
        if not isinstance(locs, (list, tuple)):
            assert isinstance(locs, Location)
            locs = [locs]
        for loc in locs:
            chains = deep_traverse(tree, loc)
            all_chains.extend(chains)
        return [[start] + chain for chain in all_chains]

    chains = []
    flow_chains_list = []

    start_loc_list = []
    from_loc_list = []
    to_loc_list = []
    for from_loc, to_loc in data_flow.items():
        from_loc_list.append(from_loc)
        to_loc_list.append(to_loc)
    for from_loc in from_loc_list:  ## 没有流向的loc，认为是数据流start_loc
        if from_loc not in to_loc_list:
            start_loc_list.append(from_loc)
    # print("len", len(start_loc_list))

    for from_loc in start_loc_list:
        flow_chains_list.extend(deep_traverse(data_flow, from_loc))
        # loc = from_loc
        # chains.append(loc)
        # while loc in data_flow:  # A->B, A->C
        #     chains.append(data_flow[loc])  # append(B)
        #     loc = data_flow[loc]  # B->
        # flow_chains_list.append(chains)
        # chains = []

    return flow_chains_list


def print_each_flow(chain_list, dataset):
    for chain in chain_list:
        for i, loc in enumerate(chain):
            if i + 1 == len(chain):
                break
            LOG.log(LOG.INFO, fmt_print(dataset[loc]))
            loc_to = chain[i + 1]
            if loc_to.record_id != loc.record_id:
                LOG.log(LOG.INFO, fmt_print(dataset[loc_to]))
            from_token = dataset[loc.filename][loc.record_id][loc.token_id]
            to_token = dataset[loc_to.filename][loc_to.record_id][loc_to.token_id]
            LOG.log(LOG.INFO, "from -> to ", from_token, "->", to_token)
            LOG.log(LOG.INFO, "\n")
        #             print(id(Location.create_location(loc.filename, loc.record_id, loc.token_id, loc.length)))
        LOG.log(LOG.INFO, "********************")


def print_flow(chain_list, dataset):
    for i, chain in enumerate(chain_list):
        record_list = []
        for loc in chain:
            record = dataset[loc]
            if record not in record_list:
                record_list.append(record)

        for record in record_list:
            LOG.log(LOG.INFO, fmt_print(record))
            LOG.log(LOG.INFO, "\n")
        LOG.log(LOG.INFO, "-------FLOW NUM is %d" % i)
        for i, loc in enumerate(chain):
            token = dataset[loc.filename][loc.record_id][loc.token_id]
            LOG.log(LOG.INFO, token, end='')
            if i + 1 < len(chain):
                LOG.log(LOG.INFO, "->", end=' ')
        LOG.log(LOG.INFO, "\n")
        for i, loc in enumerate(chain):
            token = dataset[loc.filename][loc.record_id][loc.token_id]
            LOG.log(LOG.INFO, token.loc.token_id, token, end='')
            if i + 1 < len(chain):
                LOG.log(LOG.INFO, "->", end='')
        LOG.log(LOG.INFO, "\n")
        LOG.log(LOG.INFO, "**************************************")


def build_data_flow(dataset):
    """
       给数据流加一个分类属性，这样就可以清晰的看到，这个数据流的来源
          data_flow{type_id:{from_to:to_loc}}->
          data_flow[0] = func_arg_flow_dict
          0 func_use_decl_flow_map
          1 record_local_flow_map
          2 record_inside_flow_map
          3 record_inside_equal_map
    """
    data_flow_dict = {}
    data_flow = {}

    # # func_use -> func_decl
    func_arg_flow_dict, decl_arg_map, use_arg_map, record_func_map = func_use_decl_flow(dataset)
    data_flow_dict[0] = func_arg_flow_dict
    data_flow.update(func_arg_flow_dict)

    # # func local flow def A -> local Use A
    record_local_flow_map = func_local_flow(dataset)
    data_flow_dict[1] = record_local_flow_map
    data_flow.update(record_local_flow_map)

    # # record_inside , child_decl -> father_use child_decl->contents
    record_inside_flow_map = record_inside_flow(dataset, decl_arg_map, record_func_map)
    data_flow_dict[2] = record_inside_flow_map
    data_flow.update(record_inside_flow_map)

    # record1 = dataset[".\Input\Std\MipsInstrInfo.td"][28]
    # one_record_flow = one_record_inside_flow(record1, dataset, decl_arg_map, record_func_map)
    # print()
    # print("one_record_flow\n",one_record_flow)
    # for arg_loc, token_loc in one_record_flow.items():
    #     record = dataset[arg_loc.filename][arg_loc.record_id]
    #     print(" ".join(record))
    #     record1 = dataset[token_loc.filename][token_loc.record_id]
    #     print(" ".join(record1))
    #     print(arg_loc, get_token_by_loc(arg_loc, dataset))
    #     print(token_loc, get_token_by_loc(token_loc, dataset))
    # print()

    # record_inside_equal, A=B
    record_inside_equal_map = record_inside_equal_flow(dataset, decl_arg_map, record_func_map)
    data_flow_dict[3] = record_inside_equal_map
    data_flow.update(record_inside_equal_map)

    # # print(record_inside_equal_map)
    # print(data_flow_dict)
    # for arg_loc, token_loc in data_flow.items():
    #     record = dataset[arg_loc.filename][arg_loc.record_id]
    #     print(" ".join(record))
    #     record1 = dataset[token_loc.filename][token_loc.record_id]
    #     print(" ".join(record1))
    #     print(arg_loc)
    #     print(dataset.get_tokens_by_loc(arg_loc), dataset.get_tokens_by_loc(token_loc))

    # let_constraint_flow_map = parser_let_flow(dataset)
    # data_flow_dict[4] = let_constraint_flow_map
    # data_flow.update(let_constraint_flow_map)
    #
    # print(let_constraint_flow_map)
    # print(data_flow_dict)
    # for arg_loc, token_loc in data_flow.items():
    #     record = dataset[arg_loc.filename][arg_loc.record_id]
    #     print(" ".join(record))
    #     record1 = dataset[token_loc.filename][token_loc.record_id]
    #     print(" ".join(record1))
    #     print(get_token_by_loc(arg_loc, dataset), get_token_by_loc(token_loc, dataset))

    return data_flow_dict, data_flow
