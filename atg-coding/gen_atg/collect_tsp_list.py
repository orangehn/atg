from gen_atg.utils.adjust_reference_record import normalization_tsp
from gen_atg.utils.utils import *


def collect_tsp_list(config_dict={
    "isa_mips": {
        "td": "./Input/Instruction",
    },
    "isa_ppc": {
        "td": "./Input/Other_Target/PowerPC",
    },
    "isa_arc": {
        "td": "./Input/Other_Target/ARC",
    }
}):
    super_tsp_list = []
    for isa, cfg in config_dict.items():
        reference_codes = ReferenceCode(cfg["td"])
        reference_codes.build(inst_names=None)
        for inst_name, name_value_map in reference_codes.name_value_map.items():
            for name, map in name_value_map.items():
                if name not in super_tsp_list:
                    super_tsp_list.append(name)
    super_tsp_list = normalization_tsp(super_tsp_list)
    return super_tsp_list


if __name__ == '__main__':
    config_dict = {
        "isa_mips": {
            "td": "./Input/Instruction",
        },
        "isa_ppc": {
            "td": "./Input/Other_Target/PowerPC",
        },
        "isa_arc": {
            "td": "./Input/Other_Target/ARC",
        }
    }

    super_tsp_list = collect_tsp_list(config_dict)