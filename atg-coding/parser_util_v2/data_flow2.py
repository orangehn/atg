from parser_util_v2.project_utils import LOG


def get_start_loc(flow_chains_list):
    """
    获得所有数据流开始的loc, 即用户输入参数位置
    """
    start_loc_list = []
    for chain in flow_chains_list:
        start_loc_list.append(chain[0])

    return start_loc_list


def is_syntax_function(tokens):
    if len(tokens) > 4 and tokens[0] == '(' and tokens[1] == ')' and tokens[3] == ':':   # (ins RO:$rs)
        return True
    return False


def de_repetition(dataset, flow_chains_list, data_flow_dict):
    """
    输入：flow_chains_list 是loc:loc映射
    输出：flow_dict 是str:loc映射

    目前所有start的loc里，出现重复定义的情况，应该是父类里def时给出了默认值，然后在def inst时给的输入，它们出现了重复。
    因此，需要加一个查重功能，如果参数已经给了值，是实参->形参，那么通过等号数据流分析出来的那个参数，就应该删除
    注意：此处暂时只处理两次赋值，多次赋值重复情况暂未考虑

    如果参默认值是参数位置，这里分析等号流时，应该去除参数列表里的默认值，不把它当作等号流处理，因为如果没有传入，那么它有默认值也不需要单独输入
    """
    func_use_decl_flow_map = data_flow_dict[0]
    record_local_flow_map = data_flow_dict[1]
    record_inside_flow_map = data_flow_dict[2]
    record_inside_equal_map = data_flow_dict[3]

    # start_loc_list = get_start_loc(flow_chains_list)
    """
    A<x=1>
    ADD: A<"x">
    SUB: A
    """

    name_list = []
    from_loc_dict = {}
    flow_dict = {}
    for chain in flow_chains_list:
        from_loc = chain[0]
        name_loc = chain[1]
        name = dataset.get_tokens_by_loc(name_loc)
        # name = dataset[name_loc.filename][name_loc.record_id][name_loc.token_id]
        if is_syntax_function(name): continue
        name = name[0]
        if name in name_list:
            from_1 = flow_dict[name]
            nameloc_1 = from_loc_dict[from_1]
            if from_loc in record_inside_equal_map:  # 当前数据流是等号数据流,跳过
                continue
            elif from_1 in record_inside_equal_map:  # 之前的数据流是等号数据流,from_loc直接覆盖flow_dict[name]
                flow_dict[name] = from_loc
            elif from_loc == from_1:
                continue
            else:
                print(dataset.get_record(from_loc))
                print(dataset.get_tokens_by_loc(from_loc), dataset.get_tokens_by_loc(name_loc), dataset.get_tokens_by_loc(from_1))
                if from_loc in func_use_decl_flow_map:
                    LOG.warn("1 ignore other flow type! from_loc in func_use_decl_flow_map Attention!, {}".format(name))
                    LOG.warn("{}\n {}".format(dataset[from_loc], dataset[name_loc]))
                if from_loc in record_local_flow_map:
                    LOG.warn("2 ignore other flow type! from_loc in func_use_decl_flow_map Attention!, {}".format(name))
                    LOG.warn("{}\n {}".format(dataset[from_loc], dataset[name_loc]))
                if from_loc in record_inside_flow_map:
                    LOG.warn("3 ignore other flow type! from_loc in func_use_decl_flow_map Attention!, {}".format(name))
                    LOG.warn("{}\n {}".format(dataset[from_loc], dataset[name_loc]))
                if from_1 in func_use_decl_flow_map:
                    LOG.warn("4 ignore other flow type! from_1 in func_use_decl_flow_map Attention!, {}".format(name))
                    LOG.warn("{}\n {}".format(dataset[from_1], dataset[nameloc_1]))
                    print(from_1)
                    print(nameloc_1)
                if from_1 in record_local_flow_map:
                    LOG.warn("ignore other flow type! from_1 in func_use_decl_flow_map Attention!, {}".format(name))
                    LOG.warn("{}\n {}".format(dataset[from_1], dataset[nameloc_1]))
                    print(from_1)
                    print(nameloc_1)
                if from_1 in record_inside_flow_map:
                    LOG.warn("ignore other flow type! from_1 in func_use_decl_flow_map Attention!, {}".format(name))
                    LOG.warn("{}\n {}".format(dataset[from_1], dataset[nameloc_1]))
                    print(from_1)
                    print(nameloc_1)
                print()
        else:
            from_loc_dict[from_loc] = name_loc
            name_list.append(name)
            flow_dict[name] = from_loc  # 这里存储to_name : from_loc的映射

    #     new_start_loc_list = []
    #     for name, loc in flow_dict.items():
    #         new_start_loc_list.append[loc] = name

    flow_dict1 = {}
    for name, loc in flow_dict.items():
        flow_dict1[name] = [loc]

    return flow_dict1


if __name__ == "__main__":
    pass
