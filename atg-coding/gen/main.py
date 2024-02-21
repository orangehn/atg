from gen.input_reader import read_input_data
from gen.transform import input_transform
from gen.tmplate import CommonTemplate, MyInstTemplate, FMTemplate, \
    TFFlagTemplate, EndTemplate, TypeFormatTemplate, DependenceTemplate


class AuxTemplate(object):
    def __init__(self):
        # common_template = CommonTemplate()
        self.myinst_template = MyInstTemplate('ATGInst')
        self.fm_template = FMTemplate()
        self.tfflag_template = TFFlagTemplate()
        self.dep_template = DependenceTemplate()
        self.type_format_template = TypeFormatTemplate()
        self.end_template = EndTemplate()

    def gen_shared_class(self):
        code = []
        # myinst_class = myinst_template.gen_class(super_class="Instruction")
        myinst_class = self.myinst_template.gen_class(super_class="")
        code.append(myinst_class)
        tfflag_class = self.tfflag_template.gen_class()
        code.append(tfflag_class)
        code.append(self.dep_template.gen_class())
        code.append(self.type_format_template.gen_class())
        return code

    def gen_aux_code(self, input_inst_attrs, let_attrs=[]):
        code = []
        # decl of supper class
        # code.append(common_template.gen_class(inst_attrs))
        code.append(self.fm_template.gen_class(input_inst_attrs, super_class=self.myinst_template.gen_call(input_inst_attrs)))

        # def inst with call of supper class
        call_list = [t.gen_call(input_inst_attrs) for t in [self.fm_template, self.tfflag_template]]
        # call_list.insert(0, 'Instruction')
        class_code, body = self.end_template.gen_aux_class(input_inst_attrs, call_list, let_attrs=let_attrs)
        code.append(class_code)
        # break
        return code, self.end_template.get_aux_class_name(input_inst_attrs), body

    def gen_full_code(self, input_inst_attrs, let_attrs=[]):
        code = []
        # decl of supper class
        # code.append(common_template.gen_class(inst_attrs))
        code.append(self.fm_template.gen_class(input_inst_attrs, super_class=self.myinst_template.gen_call(input_inst_attrs)))

        # def inst with call of supper class
        call_list = [t.gen_call(input_inst_attrs) for t in [self.fm_template, self.tfflag_template]]
        call_list.insert(0, 'Instruction')
        def_code = self.end_template.gen_def(input_inst_attrs, call_list, let_attrs=let_attrs)
        code.append(def_code)
        # break
        return code


def to_td(data, out_filename=None, gen_shared_class=True, let_attrs=[]):
    if out_filename is None:
        import sys
        out_file = sys.stdout
    else:
        out_file = open(out_filename, 'w')

    # common_template = CommonTemplate()
    myinst_template = MyInstTemplate('ATGInst')
    fm_template = FMTemplate()
    tfflag_template = TFFlagTemplate()
    end_template = EndTemplate()

    code = []
    if gen_shared_class:
        # myinst_class = myinst_template.gen_class(super_class="Instruction")
        myinst_class = myinst_template.gen_class(super_class="")
        code.append(myinst_class)
        tfflag_class = tfflag_template.gen_class()
        code.append(tfflag_class)

    for inst_name, inst_attrs in data.items():
        # decl of supper class
        # code.append(common_template.gen_class(inst_attrs))
        code.append(fm_template.gen_class(inst_attrs, super_class=myinst_template.gen_call(inst_attrs)))

        # def inst with call of supper class
        call_list = [t.gen_call(inst_attrs) for t in [fm_template, tfflag_template]]
        call_list.insert(0, 'Instruction')
        code.append(end_template.gen_def(inst_attrs, call_list, let_attrs=let_attrs))
        # break

    out_file.write("\n".join(code))
    # out_file.write("\n")


if __name__ == '__main__':
    data = read_input_data('./Input/inst_set_input/I/I_from_json.csv')
    input_transform(data, data_clean=False)
    to_td(data, 'I_59.td')

    aux_template = AuxTemplate()
    code = aux_template.gen_shared_class()
    for record_name in ["ADD"]:
        input_attrs = data[record_name]
        codes, aux_class_name, body = aux_template.gen_aux_code(input_attrs, [])
        code.extend(codes)
        for c in code:
            print(c)
        print("----")

        codes = aux_template.gen_full_code(input_attrs, [])
        for c in codes:
            print(c)
        break
