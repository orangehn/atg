from parser_util_v2.feature_input import *
#encoding=utf-8
def FM_Input(Feature_List):
    df = pd.read_csv('../Input/feature.csv')
    Name_FM_Mapping = {}
    for rows in df.itertuples():
        Name = getattr(rows, 'RName')
        FM = getattr(rows,'FM_Father')
        if FM != "#####" and FM not in Name_FM_Mapping.keys():
            Name_FM_Mapping[FM] = []
            for Feas in Feature_List:
                if Feas.name == Name:
                    Name_FM_Mapping[FM].append(Feas)
        elif FM != "#####" and FM in Name_FM_Mapping.keys():
            for Feas in Feature_List:
                if Feas.name == Name:
                    Name_FM_Mapping[FM].append(Feas)
    return Name_FM_Mapping

def Handle_Diff_Situation_Once(Name_FM_Mapping):
    Tem = Name_FM_Mapping.copy()
    for FMs in Tem.keys():
        for feas1 in Tem[FMs]:
            simi_ins = [feas1]
            for feas2 in Tem[FMs]:
                if feas1.name != feas2.name:
                    if IS_Similiar_Ins(feas1,feas2):
                        simi_ins.append(feas2)
            if len(simi_ins) != len(Tem[FMs]):
                tem_fm = FMs
                while tem_fm in Name_FM_Mapping.keys():
                    tem_fm += "_copy"
                Name_FM_Mapping[tem_fm] = []
                for feas3 in simi_ins:
                    Name_FM_Mapping[tem_fm].append(feas3)
                    Name_FM_Mapping[FMs].remove(feas3)
                #Debug_FM_Mapping(Name_FM_Mapping)
    return Name_FM_Mapping

def Conflict_Detect(Name_FM_Mapping):
    Tem = Name_FM_Mapping.copy()
    for FMs in Tem.keys():
        for feas1 in Tem[FMs]:
            simi_ins = [feas1]
            for feas2 in Tem[FMs]:
                if feas1.name != feas2.name:
                    if IS_Similiar_Ins(feas1, feas2):
                        simi_ins.append(feas2)
            if len(simi_ins) != len(Tem[FMs]):
                return False
    return True

def Handle_Diff_Situation(Name_FM_Mapping):
    while Conflict_Detect(Name_FM_Mapping) is False:
        Name_FM_Mapping = Handle_Diff_Situation_Once(Name_FM_Mapping)
        print(Name_FM_Mapping.keys())
    return Name_FM_Mapping


def Calculate_Arg_Number(Name_FM_Mapping):
    arg_number = {}
    for FMs in Name_FM_Mapping.keys():
        arg_number[FMs] = []
    for FMs in Name_FM_Mapping.keys():
        for feas1 in Name_FM_Mapping[FMs]:
            for feas2 in Name_FM_Mapping[FMs]:
                if feas1.name != feas2.name:
                    diff_list, need_copy = Diff_Frags(feas1,feas2)
                    if not need_copy:
                        for nums in diff_list:
                            arg_number[FMs].append(nums)
                            arg_number[FMs] = list(set(arg_number[FMs]))
    return arg_number

def Generate_Bits_Codes(start_bit, end_bit):
    if start_bit == end_bit:
        return "bit"
    else:
        return "bits<{}> ".format(abs(start_bit - end_bit) + 1)

def Generate_Class_Codes(FM_Name, Arg_List, Feature):
    class_codes = "Class " + FM_Name
    if Arg_List != []:
        class_codes += " < "
    for args in Arg_List:
        if not Feature.frags[args].is_split:
            class_codes += Generate_Bits_Codes(Feature.frags[args].start_bit, Feature.frags[args].end_bit)
            class_codes += Feature.frags[args].name
            class_codes += " ,"
    if Arg_List != []:
        class_codes = class_codes[0:-1] + ">"
    print(class_codes + "\n")
    return class_codes + "\n"

def Generate_Let_Frag(ins_start_bit, ins_end_bit, frag_start_bit, frag_end_bit, name, is_split):
    if is_split:
        if frag_end_bit == frag_start_bit:
            print("let Inst{{{}}} = {}{{{}}};\n".format(ins_start_bit, name, frag_start_bit))
            return "let Inst{{}} = {}{{}};\n".format(ins_start_bit, name, frag_start_bit)
        else:
            print("let Inst{{{}-{}}} = {}{{{}-{}}};\n".format(ins_start_bit, ins_end_bit, name, frag_start_bit, frag_end_bit))
            return "let Inst{{{}-{}}} = {}{{{}-{}}};\n".format(ins_start_bit, ins_end_bit, name, frag_start_bit, frag_end_bit)
    else:
        if ins_start_bit == ins_end_bit:
            print("let Inst{{}} = {};\n".format(ins_start_bit, name))
            return "let Inst{{}} = {};\n".format(ins_start_bit, name)
        else:
            print("let Inst{{{}-{}}} = {};\n".format(ins_start_bit,ins_end_bit,name))
            return "let Inst{{{}-{}}} = {};\n".format(ins_start_bit,ins_end_bit,name)

def Generate_Let_Codes(Feature):
    let_codes = ""
    for frag in Feature.frags:
        if not frag.is_split:
            let_codes += Generate_Let_Frag(frag.start_bit,frag.end_bit,0, 0, frag.name, False)
        else:
            for small_frags in frag.frags_split:
                let_codes += Generate_Let_Frag(small_frags.ins_start_bit, small_frags.ins_end_bit, small_frags.frag_start_bit, small_frags.frag_end_bit, frag.name, True)
    return  let_codes



def Generate_Def_Value(Feature, frag):
    value_codes = ""
    if frag.Value:
        value_codes += " = 0b{}".format(frag.Value)
    return value_codes

def Generate_Def_Codes(Feature, Arg_List):
    def_codes = ""
    arg_name = []
    for args in Arg_List[Feature.FM]:
        arg_name.append(Feature.frags[args].name)
    def_codes += "bits<{}> Inst;\n".format(Feature.Size * 8)
    print("bits<{}> Inst;\n".format(Feature.Size * 8))
    for frag in Feature.frags:
        if frag.name not in arg_name:
            if not frag.is_split:
                if abs(frag.start_bit - frag.end_bit) + 1 == 1:
                    def_codes += "bit {}".format(frag.name)
                    def_codes += Generate_Def_Value(Feature,frag)
                    def_codes += ";\n"
                    print("bit {}".format(frag.name) + Generate_Def_Value(Feature,frag) + ";\n")
                else:
                    def_codes += "bits<{}> {}".format((abs(frag.start_bit - frag.end_bit) + 1), frag.name)
                    def_codes += Generate_Def_Value(Feature, frag)
                    def_codes += ";\n"
                    print("bits<{}> {}".format((abs(frag.start_bit - frag.end_bit) + 1), frag.name) + Generate_Def_Value(Feature,frag) + ";\n")
            else:
                total_len = 0
                for small_frags in frag.frags_split:
                    total_len += (abs(small_frags.frag_start_bit - small_frags.frag_end_bit) + 1)
                def_codes += "bits<{}> {};\n".format(total_len, frag.name)
                print("bits<{}> {};\n".format(total_len, frag.name))
    return def_codes

def Generate_FM_Codes(Name_FM_Mapping, Arg_Number):
    for FMs in Name_FM_Mapping.keys():
        Generate_Class_Codes(FMs,Arg_Number[FMs],Name_FM_Mapping[FMs][0])
        print("{")
        Generate_Def_Codes(Name_FM_Mapping[FMs][0], Arg_Number)
        Generate_Let_Codes(Name_FM_Mapping[FMs][0])
        print("}")
        #for frags in []:
        #    pass


def Debug_FM_Mapping(Name_FM_Mapping):
    print("##################################")
    for FMs in Name_FM_Mapping.keys():
        print(FMs + ":  [", end = "")
        for feas in Name_FM_Mapping[FMs]:
            print(feas.name + ",", end = "")
        print("]")
    print("##################################")

if __name__ == "__main__":
    F_List = Feature_Construction()
    Name_FM_Mapping = FM_Input(F_List)
    Name_FM_Mapping = Handle_Diff_Situation(Name_FM_Mapping)
    Debug_FM_Mapping(Name_FM_Mapping)
    Arg_Number = Calculate_Arg_Number(Name_FM_Mapping)
    Generate_FM_Codes(Name_FM_Mapping, Arg_Number)