import csv

from gen.input_reader import read_input_data
from gen.transform import input_transform
from gen.main import to_td, AuxTemplate
from gen_atg.utils.adjust_reference_record import instr_dict
from gen_atg.utils.utils import *


def transform(data):
    pass


class CodeGen(object):

    def __init__(self, reference_codes: ReferenceCode):
        """
            templates: dict(inst_name=list[record]), template of reference code, like MIPS
        """
        self.reference_codes = reference_codes
        self.aux_template = AuxTemplate()
        self.code_merger = CodeMerger()

    def match_reference_code(self, inst_name, inst_attrs):
        if 'matched_inst' not in inst_attrs:
            return None
        reference_inst_name = inst_attrs['matched_inst']
        return reference_inst_name

    def gen_from_aux_template(self, inst_attrs):
        return self.aux_template.gen_full_code(inst_attrs)

    def merge_with_aux_template(self, relation_asts, inst_name, inst_attrs):
        """
            def ADDX: xxx
            inst_name: 'ADD'
        =>
            class ADD_AUX: xxxx
            def ADD: xxxx, ADD_AUX{
                let xxx;
                let xxx;
            }
        """
        codes, aux_class_name, body = self.aux_template.gen_aux_code(inst_attrs)
        relation_asts[inst_name].append_super_class(aux_class_name)
        relation_asts[inst_name].append_content_str(body)
        return relation_asts, codes

    def merge_and_split(self, gen_codes):
        return self.code_merger.run(gen_codes)

    def out_to_td(self, gen_code, save_path):
        import sys, os
        save_dir, _ = os.path.split(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        stdout = sys.stdout
        sys.stdout = open(save_path, 'w')

        for code in gen_code:
            print(str(code))

        sys.stdout.close()
        sys.stdout = stdout

    def run(self, data, save_path, gen_shared_code=True):
        transform(data)

        aux_codes = []
        if gen_shared_code:
            aux_codes.extend(self.aux_template.gen_shared_class())

        gen_codes = {}
        for inst_name in data:
            input_inst_attrs = data[inst_name]
            reference_inst_name = self.match_reference_code(inst_name, input_inst_attrs)
            if reference_inst_name is None:
                aux_codes.extend(self.gen_from_aux_template(input_inst_attrs))
                continue
            rc, name_value_map = self.reference_codes.get_reference_code(reference_inst_name)
            rc, new_input_inst_attrs = self.reference_codes.replace_with_kv_and_rename(
                rc, name_value_map, input_inst_attrs, reference_inst_name, inst_name)

            rc, aux_code = self.merge_with_aux_template(rc, inst_name, new_input_inst_attrs)
            gen_codes[inst_name] = rc
            aux_codes.extend(aux_code)
        gen_codes = self.merge_and_split(gen_codes)
        codes = aux_codes + gen_codes
        self.out_to_td(codes, save_path)


def add_match_data(data, csv_file):
    row_data = []
    f = open(csv_file)
    reader = csv.reader(f)
    for row in reader:
        row_data.append(row)
    f.close()
    title = row_data[0]
    # riscv_name_idx = title.index('riscv_name')
    rname_upper_idx = title.index('rname_upper')
    mips_name_idx = title.index('root_name')

    # rname2rname_upper = {}
    # for rname_upper, inst_data in data.items():
    #     rname2rname_upper[inst_data['rname']] = rname_upper

    ignore_rname_upper = []
    for i, row in enumerate(row_data[1:]):
        mips_name = row[mips_name_idx].strip()[4:]  # remove "def " in "def ADD"
        rname_upper = row[rname_upper_idx]
        if rname_upper not in data:
            ignore_rname_upper.append((i+2, rname_upper))
            continue
        # riscv_upper_name = rname2rname_upper[riscv_lower_name]
        data[rname_upper]['matched_inst'] = mips_name
    assert len(ignore_rname_upper) == 0, ignore_rname_upper



if __name__ == '__main__':
    config_dict = {
        inst_set: {
            # "td": "./Input/Instruction",
            # Input/inst_set_input/{inst_set}/{inst_set}_from_json.csv
            "input_files": [
                f"./Input/inst_set_input/{inst_set}/{inst_set}_v1.csv",
                f"./Input/inst_set_input/{inst_set}/{inst_set}_inst_match.csv", ""],
            "out": f"./output/inst_set/{inst_set}_inst.td"
        } for inst_set in ["I", "m", "a", "b", "c", "d", "f", "zfh", "v"]}

    # config_dict = {
    #     inst_set: {
    #         # "td": "./Input/Instruction",
    #         # Input/inst_set_input/{inst_set}/{inst_set}_from_json.csv
    #         "input_files": [f"./Input/inst_set_input_32/{inst_set}/{inst_set}_32_v1.csv",
    #                         f"./Input/inst_set_input_32/{inst_set}/{inst_set}_inst_match_32.csv", ""],
    #         "out": f"./output/inst_set_32/{inst_set}_inst_32.td"
    #     } for inst_set in ["I", "m", "a", "b", "c", "d", "f", "zfh"]}

    config_dict.update({
        # "test": {
        #     "td": "unit_test/tmp.td",
        # }
    })

    reference_codes = ReferenceCode("./Input/Instruction")
    del_tsp = []
    reference_codes.build(inst_names=None)
    for inst_set, inst_cfg in config_dict.items():
        print(inst_set)
        data = read_input_data(inst_cfg['input_files'][0])
        add_match_data(data, inst_cfg['input_files'][1])
        input_transform(data, data_clean=False)
        # data.keys
        for inst, kv in data.items():
            for k, v in kv.items():
                if v is not None and k in del_tsp:
                    print(inst_set, inst, k)
        code_gen = CodeGen(reference_codes)
        code_gen.run(data, inst_cfg['out'], gen_shared_code=inst_set == 'I')
