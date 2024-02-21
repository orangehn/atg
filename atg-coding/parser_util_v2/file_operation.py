import copy
from collections import defaultdict

from parser_util_v2.data_flow import *
# from parser_util_v2.get_relation_set import *
from parser_util_v2.project_utils import *
from parser_util_v2.dataset_constructor import readlines, find_token, SuperToken

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


def get_relation_record(relation_set, dataset):
    """
    get records from dataset by record_id set
    """
    relation_records = []
    for loc in relation_set:
        r = dataset[loc.filename][loc.record_id]
        relation_records.append(copy.copy(r))
    # new_relation_records = copy.deepcopy(relation_records)
    return relation_records


# def single_instruction_deal(dataset, inst_name, global_def_table, flow_dict):
#     relation_set = get_relation_set(dataset, [inst_name], global_def_table)
#     #for record in relation_records:
#     #    print(record)
#     relation_records = get_relation_record(relation_set, dataset)
#
#     # # 5.1 print start_loc and name of dataflow
#     # rt_data_flow = get_rt_flow(relation_records, data_flow, global_def_table, dataset)  # get flow of relation set
#     # flow_chains_list = get_flow_chain(rt_data_flow)
#     # for chain in flow_chains_list:
#     #     for c in chain:
#     #         # print(c, end='->')
#     #         print(dataset.get_token_by_loc(c), end='->')
#     #     print()
#     # sum = 0
#     # for chain in flow_chains_list:
#     #     loc = chain[0]
#     #     token = dataset[loc.filename][loc.record_id][loc.token_id]
#     #     #     if token.isdigit() or token.startswith('"'):
#     #     sum += 1
#     #     name_loc = chain[1]
#     #     name = dataset[name_loc.filename][name_loc.record_id][name_loc.token_id]
#     #     print(name, "=", token)
#     # print(sum)
#
#     # # 5.2 remove default value when input given
#     # flow_dict = de_repetition(dataset, flow_chains_list, data_flow_dict)
#     flow_dict.update(parser_bits_pattern(relation_records, dataset))
#     return flow_dict, relation_set


def parser_bits_pattern(records, dataset, link_sym='_'):
    bits_flow = {}
    for r in records:
        for var_type, var_name, var_value in r.data['args'] + r.data['content']:
            idx = find_token(var_type, 0, 'bits')
            if idx >= 0:
                size = var_type[idx + 1]
                assert len(size) == 3 and SuperToken.is_super_token(size, '<', '>'), size
                assert len(var_name) == 1, var_name
                bits_flow["bits" + link_sym + var_name[0]] = (size[1].loc,)
            for v in var_name:
                if SuperToken.is_super_token(v, '{', '}'):
                    # DForm_1,   let Inst{11-15} = Addr{20-16}; // Base Reg
                    if (len(var_value) == 2 and SuperToken.is_super_token(var_value[1], '{', '}')) or \
                            (len(var_value) == 3 and var_value[1] == '.'):
                        break
                    assert len(var_name) == 2 and len(var_value) == 1, f"{var_name} {var_value}"
                    assert len(v) == 3 or len(v) == 5, v
                    name = var_name[0].new("{}{}{}".format(var_name[0], link_sym, var_value[0]))
                    value = (v[1].loc,) if len(v) == 3 else (v[1].loc, v[3].loc)
                    bits_flow[name] = value
                    break
    return bits_flow


def get_inst_format_args(records, G):
    inst_format_records = set()
    for r in records:
        for t, n, v in r.data['content']:
            if n[0] in ['Inst', 'BI', 'BIBI'] and v is not None and v[0] != 'Opcode' and v[0] != 'opcode':
                inst_format_records.add(n[0].loc)
                break
    # if len(inst_format_records) != 1:
    #     for r in inst_format_records:
    #         print(G.dataset.get_record(r))
    #     print()
    #     raise ValueError
    # print("************", list(inst_format_records))
    if inst_format_records:
        inst_format_record_loc = list(inst_format_records)[0]
        return inst_format_record_loc.filename, inst_format_record_loc.record_id
    else:
        return None


def single_instruction_deal(dataset, inst_name, G, name_value_loc):
    relation_set = G.get_relation_set([inst_name])
    relation_records = get_relation_record(relation_set, dataset)

    inst_format_record_fig = get_inst_format_args(relation_records, G)

    relation_nv_loc, relation_infos = G.get_name_value_loc_of_records(name_value_loc, relation_records, inst_name)
    relation_nv_loc.update(parser_bits_pattern(relation_records, dataset))
    return relation_nv_loc, relation_set, inst_format_record_fig, relation_infos


def get_all_flow_dict(dataset, inst_names, G, name_value_loc):
    all_flow_dict = {}
    inst_format_record_fig = {}
    relation_id_dict = {}
    relation_infos = {}
    # relation_sets = set()
    for inst_name in tqdm(inst_names):
        # print("solving inst", inst_name)
        flow_dict, relation_set, inst_format_record_fig[inst_name], relation_infos[inst_name] = single_instruction_deal(
            dataset, inst_name, G, name_value_loc)

        all_flow_dict[inst_name] = flow_dict
        relation_id_dict[inst_name] = relation_set
        # relation_sets = relation_sets | relation_set
    # relation_records = get_relation_record(relation_sets, dataset)
    return all_flow_dict, relation_id_dict, inst_format_record_fig, relation_infos


def read_def_inst(filename):
    inst_names = []
    riscv_inst_names = []
    for line in readlines(filename)[1:]:
        if line == '\n':
            continue
        line_data = line.split(',')
        x = line_data[0].split(' ')
        inst_names.append(x[1].strip())
        # riscv_inst_names.append(line_data[0])
    LOG.debug("(main)", "inst_names", inst_names)
    # LOG.debug("(riscv)", "inst_names", riscv_inst_names)
    return inst_names  # , riscv_inst_names


def read_def_inst2(filename):
    inst_names = []
    # riscv_inst_names = []
    for line in readlines(filename)[1:]:
        if line == '\n':
            continue
        line_data = line.split('\n')
        # x = line_data[0].split(' ')
        # inst_names.append(x[0].strip())
        inst_names.append(line_data[0])
    # LOG.debug("(main)", "inst_names", inst_names)
    # LOG.debug("(riscv)", "inst_names", riscv_inst_names)
    return inst_names  # , riscv_inst_names


import sys


def sort_dict(data):
    return sorted(data, key=lambda x: x[0])


def turn_name_to_instrfield(name):
    if name == "asmstr":
        name = "AsmString"
    if name == "OOL":  # "outs":
        name = "OutOperandList"
    if name == "IOL":  # ins":
        name = "InOperandList"
    if name == "pattern":
        name = "Pattern"
    if name == "isComm":
        name = "isCommutable"
    if name == "Namespace":
        name = "ISA_Name"
    if name == "Arch":
        name = "Arch_Edition"
    # if name == "IsComm":
    #     name = ""
    if name == "Uses":
        name = "UseRegs"
    if name == "Defs":
        name = "DefRegs"
    if name == "C":
        name = "Constraints"
    if name == "E":
        name = "DisableEncoding"
    return name


def sep_inst_format_args(dataset, all_flow_dict, inst_format_record_fig):
    inst_format_args = defaultdict(dict)
    other_flow_dict = defaultdict(dict)
    for inst_name, flow_dict in all_flow_dict.items():
        for name, locs in flow_dict.items():
            loc = locs[0]
            if (loc.filename, loc.record_id) == inst_format_record_fig[inst_name]:
                # str = dataset.get_tokens_by_loc(loc)
                # print(name)
                if name.startswith("bit") or name.startswith("Inst") or name.startswith("BI") or \
                        name.startswith("BIBO"):
                    inst_format_args[inst_name][name] = locs
            else:
                other_flow_dict[inst_name][name] = locs
    return other_flow_dict, inst_format_args


def is_redundant(name, loc_str):
    flag = False
    if name in ["Predicates", "BaseOpcode", "TSFlags", "Itinerary", "OpNode", "RO", "Operation", "funct",
                "Inst", "bits_TSFlags", "bits_Inst", "Opcode", "operator", "Addr", "shift", "Op",
                "cond", "bits_fmt", "bits_c", "bits_fcc", "Format", "vt", "rs", "shamt", "bits_rs", "bits_rt",
                "bits_imm", "Inst_rs", "Inst_rt", "Inst_imm", "fcc", "bits_shift", "bits_Inst",
                "bits_fmt", "bits_c", "bits_fcc"]:# "SoftFail",]:
        flag = True
    if loc_str in ["null_frag"]:
        flag = True
    return flag


def need_change(name):
    if name in ["BaseOpcode+instr_asm", "opstr+BaseOpcode", "opstr", "instr_asm", "BaseOpcode+opstr+BaseOpcode",
                "BaseOpcode+AsmString", "asmstr+BaseOpcode", "asmstr", "AsmString+BaseOpcode"]:
        name = "BaseOpcode+opstr"
    if name in ["typestr"]:
        name = "Typestr"
    if name == "n": # for reg
        name = "AsmName"
    return name


def Normalization_cf(dataset, all_flow_dict, op_def, patnode, inst_format_record_fig: dict):
    all_flow_dict, inst_format_args = sep_inst_format_args(dataset, all_flow_dict, inst_format_record_fig)
    # for inst_name, all_data in sorted(inst_format_args.items(), key=lambda x: x[0]):
    #     print(inst_name, ":************************************************************")
    #     for name, values in all_data.items():
    #         print("\t", name, ":", [dataset.get_tokens_by_loc(value) for value in values])
    # print("END **************************************************************************")

    new_all_flow_dict = {}

    for inst_name, flow_dict in all_flow_dict.items():
        new_flow_dict = {}
        for name, loc in flow_dict.items():
            # loc = loc[0]
            loc_str = ",".join([" ".join(dataset.get_tokens_by_loc(l)) for l in loc])
            # if name == "cond_op":
                # print(loc_str)
                # print(patnode)
                # print(is_redundant(name, loc_str), loc_str in patnode, loc_str in op_def)
            if is_redundant(name, loc_str) or loc_str in patnode or loc_str in op_def:
                # print("name, catch loc_str:", name, loc_str)
                continue
            name = need_change(name)
            new_flow_dict[name] = loc
        new_all_flow_dict[inst_name] = new_flow_dict
    # print_input(dataset, new_all_flow_dict)

    return new_all_flow_dict


def write_cp2file(filename, inst_input_dict):
    file_handle = open("./output/hot_pic/" + filename, mode='w')
    for inst_name, all_data in inst_input_dict.items():
        input_names = []
        for key, values in all_data.items():
            input_names.extend(values)
        file_handle.write(inst_name + "," + ",".join(input_names) + "\n")
    file_handle.close()

def print_input1(dataset, all_flow_dict, instr_fields, op_def, patnode, inst_format_record_fig: dict, func_type):
    all_flow_dict, inst_format_args = sep_inst_format_args(dataset, all_flow_dict, inst_format_record_fig)
    # for inst_name, all_data in sorted(inst_format_args.items(), key=lambda x: x[0]):
    #     print(inst_name, ":************************************************************")
    #     for name, values in all_data.items():
    #         print("\t", name, ":", [dataset.get_tokens_by_loc(value) for value in values])
    # print("END **************************************************************************")

    inst_input_dict = {}
    # stdout = sys.stdout
    # sys.stdout = open("input_type.csv", 'w')
    inst_des = []
    inst_des1 = ["op", "opcode", "funct", "opstr", "rs", "rt", "Inst", "offset", "code_1", "code_2", "addr",
                 "stype", "rotate", "shamt", "op", "opcode", "opstr", "Inst", "offset", "BO", "A", "L", "B","bi",
                 "aa", "RC", "xo", "lk", "PPC64", "Addr","Predicates", "BaseOpcode", "TSFlags", "Itinerary", "SoftFail",
                 "OpNode", "RO", "Operation", "funct","Inst", "bits_TSFlags", "bits_SoftFail", "bits_Inst", "Opcode",
                 "operator", "Addr", "shift", "Op", "BH","RB", "CR", "oe", "bo",
                 "cond", "bits_fmt", "bits_c", "bits_fcc", "Format", "vt", "rs", "shamt", "bits_rs", "bits_rt",
                 "bits_imm", "Inst_rs", "Inst_rt", "Inst_imm", "fcc", "bits_shift", "bits_SoftFail", "bits_Inst",
                 "bits_fmt", "bits_c", "bits_fcc"]
    op_list = ["UseRegs", "DefRegs"]
    tsflags = ["hasForbiddenSlot", "IsPCRelativeLoad", "hasFCCRegOperand","PPC970_First", "PPC970_Single",
               "PPC970_Cracked", "PPC970_Unit", "XFormMemOp"]
    for str in inst_des1:
        inst_des.append(str.lower())

    for inst_name, flow_dict in all_flow_dict.items():
        all_data = defaultdict(list)
        for name, loc in flow_dict.items():
            loc = loc[0]
            if isinstance(loc, tuple):
                loc_str = ",".join([" ".join(dataset.get_tokens_by_loc(l)) for l in loc])
            else:
                loc_str = " ".join(dataset.get_tokens_by_loc(loc))
            name = turn_name_to_instrfield(name)
            # if name == "sz":
            #     name = "Size"
            # if name in ["SoftFail", "bits-SoftFail", "itin", "Itin", "Itinerary", "vt", "shift", "_funct", "funct2",
            #             "guest", "mfmt", "OpNode"]:
                # if name in ["SoftFail", "bits-SoftFail", "itin", "Itin", "Itinerary", "subop", "mincode", "opasm", "F",
                #             "major", "BaseOpcode", "b16", "IsU6", "pat"]:
            if name in ["SoftFail", "bits_SoftFail", "itin", "Itin", "Itinerary", "asmbase"]:
                all_data["Delete"].append(name)
            elif is_instruction_field(name, instr_fields):
                all_data["instruction"].append(name)
            elif name in ["bits_SoftFail"]:
                all_data["high_level"].append(name)
            elif name in ["bits_FormBits", "bits_Value", "bits_val", "FormBits", "val"]:
                all_data["inst_type"].append(name)
            # elif name.startswith("Inst-") or name.lower() in inst_des or name.startswith("bits-") or name in ["rd"] \
            #         or name.startswith("BI") or name.startswith("BIBI"):
            elif name.lower().startswith("inst") or name in inst_des1 or name.lower().startswith("bits_") or \
                    name.lower().startswith("bi") or name.lower().startswith("bibi"):
                all_data["inst_format"].append(name)
            elif name.lower().startswith("tsflags") or name in tsflags:
                all_data["TSFlags"].append(name)
            elif name.lower().endswith("predicates") or name.lower().endswith("predicate"):
                all_data["Delete"].append(name)
            elif name in op_list:
                all_data["Register"].append(name)
            elif name in ["BaseName", "Interpretation64Bit"]:
                all_data["RowFields"].append(name)
            elif loc_str in op_def or loc_str in patnode:
                continue
            else:
                all_data["high_level"].append(name)
        # print("342**********", inst_name, all_data["high_level"])
        all_data["inst_format"] = ["Bit_Encoding[Name]", "Bit_Encoding[Range]", "Bit_Encoding[Type]",
                                   "Bit_Encoding[Value]", "Operand[Type]", "Operand[Precision]", "Operand[Illegal]"]
        all_data["TSFlags"] = ["Target_Specific_Flags"]
        all_data["Delete"] = []
        all_data["inst_type"] = ["Format_Classification"]
        all_data["RowFields"] = ["InstrMapping"]
        inst_input_dict[inst_name] = all_data
    # for inst_name, all_data in sorted(inst_input_dict.items(), key=lambda x: x[0]):
    #     print(inst_name, ":************************************************************")
    #     for key, values in all_data.items():
    #         print("\t", key, ":")
    #         for name in values:
    #             print("\t\t", name)
    # sys.stdout = stdout
    Mips32_dict = {}
    Mips64_dict = {}
    MipsFPU_dict = {}
    MicroMips_dict = {}
    Mips32r6_dict = {}
    Mips64r6_dict = {}
    MipsMAS_dict = {}
    ARC_dict = {}
    PPC_dict = {}
    for inst_name, all_data in inst_input_dict.items():
        filename = inst_name.loc.filename
        if filename.endswith("MicroMipsInstrInfo.td" ):
            MicroMips_dict[inst_name] = all_data
        elif filename.endswith("Mips64InstrInfo.td" ):
            Mips64_dict[inst_name] = all_data
        elif filename.endswith("MipsInstrFPU.td" ):
            MipsFPU_dict[inst_name] = all_data
        elif filename.endswith("MipsInstrInfo.td" ):
            Mips32_dict[inst_name] = all_data
        elif filename.endswith("Mips32r6InstrInfo.td" ):
            Mips32r6_dict[inst_name] = all_data
        elif filename.endswith("Mips64r6InstrInfo.td" ):
            Mips64r6_dict[inst_name] = all_data
        elif filename.endswith("MipsMSAInstrInfo.td" ):
            MipsMAS_dict[inst_name] = all_data
        elif filename.endswith("ARCInstrInfo.td" ):
            ARC_dict[inst_name] = all_data
        elif filename.endswith("PPCInstrInfo.td" ):
            PPC_dict[inst_name] = all_data

    if func_type == "Instruction":
        write_cp2file("Mips32.csv", Mips32_dict)
        write_cp2file("Mips64.csv", Mips64_dict)
        write_cp2file("MipsFPU.csv", MipsFPU_dict)
        write_cp2file("MicroMips.csv", MicroMips_dict)
        write_cp2file("Mips32r6.csv", Mips32r6_dict)
        write_cp2file("Mips64r6.csv", Mips64r6_dict)
        write_cp2file("MipsMSA.csv", MipsMAS_dict)
    if func_type == "ARC":
        write_cp2file("ARC.csv", ARC_dict)
    if func_type == "PPC":
        write_cp2file("PPC.csv", PPC_dict)


def print_input(dataset, all_flow_dict):
    # stdout = sys.stdout
    # sys.stdout = open("inst_flow.csv", 'w')
    i = 0
    for inst_name, flow_dict in all_flow_dict.items():
        if i > 599:
            print("inst:", inst_name)
            for name, loc in flow_dict.items():
                loc_str = ",".join([" ".join(dataset.get_tokens_by_loc(l)) for l in loc])
                print(loc[0], name, loc_str)
            LOG.info("**********************************************************")
        i = i+1
    # sys.stdout = stdout


def write_input(dataset, all_flow_dict):
    file_handle = open("input.csv", mode='w')
    for inst_name, flow_dict in all_flow_dict.items():
        names_list = flow_dict.keys()
        name_list = []
        for name in names_list:
            # if is_special_arg(name):
            #     continue
            name_list.append(name)
        file_handle.write(inst_name + "," + ",".join(name_list) + "\n")
    file_handle.close()


def write_input_count(all_flow_dict):
    file_handle = open("input_count.csv", mode='w')
    for inst_name, flow_dict in all_flow_dict.items():
        names_list = flow_dict.keys()
        name_list = []
        for name in names_list:
            # if is_special_arg(name):
            #     continue
            name_list.append(name)
        sum = len(name_list)
        file_handle.write(inst_name + "," + str(sum) + "\n")
    file_handle.close()


def is_instruction_field(name, instr_fields):
    return name in instr_fields


def is_special_arg(name):
    format_str = ["opstr", "op", "funct", "val", "outs", "ins", "asmstr", "pattern", "rd", "rs", "rt", "f", "value",
                  "Opcode", "FormBits", "Inst", "Inst-Opcode", "Inst-op", "Inst-rs", "Inst-rt", "Inst-rd", "Inst-0",
                  "Inst-funct", "imm16", "Inst-imm16", "instr_asm", "BaseOpcode", "OutOperandList", "InOperandList",
                  "AsmString", "Pattern", "offset", "Inst-offset", "code_1", "code_2", "Inst-0x0", "Inst-code_1",
                  "Inst-code_2", "MO", "addr", "Inst-0b000001", "Inst-0b11111", "stype", "Inst-stype", "Inst-0xf",
                  "target", "Inst-target", "Inst-9", "Inst-addr", "LLBit", "Inst-0x10", "Inst-1", "Inst-LLBit",
                  "rotate",
                  "Inst-rotate", "shamt", "Inst-shamt"]
    special_str = ["OpNode", "RO", "Od", "imm_type", "GPROpnd", "opnd", "cond_op", "Defs", "Imm", "ImmOpnd", "PF"]
    default_str = ["Itin", "itin", "Addr", ]
    # cplus_str = ["DecoderMethod", "EncodingPredicates", "InsnPredicates"]
    return name in format_str or name in special_str or name in default_str #or name in cplus_str


def get_super_input(dataset, all_flow_dict):
    super_input_list = []  # {}
    for inst_name, flow_dict in all_flow_dict.items():
        for name, loc in flow_dict.items():
            # length = loc.length
            # end_id = loc.token_id + length - 1
            # token = dataset[loc.filename][loc.record_id][loc.token_id:end_id + 1]
            if name not in super_input_list:  # and not is_special_arg(name):
                super_input_list.append(name)
                # arg = (loc, token)
                # super_input_list[name] = arg
    return super_input_list


def mark_color(all_flow_dict, relation_id_dict, dataset):
    for inst_name in all_flow_dict.keys():
        nv_loc, relation_ids = all_flow_dict[inst_name], relation_id_dict[inst_name]

        record_names = [dataset.get_record(loc).data['name'] for loc in relation_ids]
        # relation_trg = self.token_ref_graph.sub_graph_of_records(record_names)
        # start_locs = relation_trg.get_start_use_locations()

        for locs in nv_loc.values():
            for loc in locs:
                for t in dataset.get_tokens_by_loc(loc):
                    t.color = 'CP'

        keywords = {'def', 'class', 'let', 'foreach', 'multiclass', 'defm', 'bits', 'string', 'bit', 'dag', 'list',
                    'listconcat', 'strconcat'}
        for rname in record_names:
            r = dataset.records_map[rname]
            for t in r:
                import string
                if t in string.punctuation:
                    t.color = 'LS'
                elif t in keywords:  # keyword set as 'LS'
                    t.color = 'LS'
                elif t in dataset.records_map:  # class name set as 'PN'
                    t.color = 'PN'
                elif hasattr(t, 'color') and t.color == 'CP':  # value location set as 'CP'
                    continue
                elif not t.isidentifier():  # symbol set as 'X'
                    t.color = 'LS'
                else:
                    t.color = 'PN'

    def statistic_color(dataset):
        colors = {}
        for rname, r in dataset.records_map.items():

            for t in r:
                if t.color not in colors:
                    colors[t.color] = 0
                else:
                    colors[t.color] += 1
        ss = sum(colors.values())
        for c, v in colors.items():
            print(c, v, v / ss)

    statistic_color(dataset)


def print_with_color(nv_loc, relation_ids, dataset):
    record_names = [dataset.get_record(loc).data['name'] for loc in relation_ids]
    # relation_trg = self.token_ref_graph.sub_graph_of_records(record_names)
    # start_locs = relation_trg.get_start_use_locations()

    color_map = {
        'CP': '\033[1;31;40m',  # red, see https://zhuanlan.zhihu.com/p/150005254
        'PN': '\033[1;32;40m',  # green
        'LS': '\033[1;34;40m',   # blue
    }
    for rname in record_names:
        r = dataset.records_map[rname]
        ts, colors = [], []
        for t in r:
            l = max(len(t), len(t.color))
            ts.append(' ' * (l - len(t)) + t)
            colors.append(' ' * (l-len(t.color)) + t.color)
        # print(' '.join(ts))
        print(' '.join([color_map[c.strip()]+t for t, c in zip(ts, colors)]))
        print('\033[0m', end='')
        print(' '.join(colors))
