from parser_util_v2.dataset_type import Token, Record, VarToken, SuperToken, TokenPos
from collections import defaultdict, namedtuple
from copy import deepcopy
from parser_util_v2.project_utils import LOG, Flag
from third_utils.graph_utils import SimpleGraph, topology_sort


def extend_records_dict(records_dict: dict, extend_data: dict):
    for k in extend_data:
        records_dict[k].extend(extend_data[k])


def find_end_token(srecord):
    final_idx = -1
    for i in range(len(srecord) - 1, -1, -1):
        if isinstance(srecord[i], SuperToken) or (not srecord[i].isspace()):
            final_idx = i
            break
    assert final_idx >= 0, "last char must be find in {}".format(srecord)
    return final_idx


def remove_token(record, token):
    r = [t for t in record if t != token]
    if isinstance(record, Record):
        r = record.new(r)
    return r


def find_token(tokens, start_idx, key_token):
    """
    查看tokens中是否出现了key_tokens中的token,出现了就返回该token位置
    """
    for idx in range(start_idx, len(tokens)):
        if tokens[idx] == key_token:
            return idx
    return -1


def get_first_token(token_list):
    if isinstance(token_list, Token):
        return token_list
    elif isinstance(token_list, list):
        return get_first_token(token_list[0])
    else:
        raise TypeError()


def get_last_token(token_list):
    if isinstance(token_list, Token):
        return token_list
    elif isinstance(token_list, list):
        return get_first_token(token_list[-1])
    else:
        raise TypeError()


class Pass(object):
    @staticmethod
    def do(records_dict: dict, *args):
        raise NotImplementedError


# single record pass

class RecordSplitPass(Pass):
    @staticmethod
    def do(token_list, one_line_key=["include"], is_super_token=False, return_super_token=False, *args):
        """
        class/def
        先转成super_token在处理
        """
        if is_super_token:
            stoken_list = token_list
        else:  # 转换成supertoken list
            stoken_list = SuperToken.collect_super_token(token_list)

        record_list = []
        idx = 0
        while idx < len(stoken_list):
            token = stoken_list[idx]
            if token in one_line_key:  # include
                last_idx = find_token(stoken_list, idx + 1, '\n')
                if last_idx == -1:  # file end
                    last_idx = len(stoken_list)-1
                record = stoken_list[idx: last_idx+1]
                idx = last_idx+1
            elif token in ["def", "class", "multiclass", "defm", "defset"]:
                record, idx = RecordSplitPass.split_by_record_def(stoken_list, idx)
            elif token in ['let', 'foreach']:
                record, idx = RecordSplitPass.split_by_record_let(stoken_list, idx)
            elif token == '\n':
                idx += 1
                continue
            else:
                raise ValueError(
                    "ignore ({})(line {}) in ({})".format(token, token.line_id, stoken_list[idx - 10: idx + 10]))

            record = remove_token(record, '\n')
            record = token_list.new(record) if isinstance(token_list, Record) else Record(record)
            record.type = record[0]

            if not return_super_token:
                record = SuperToken.unfold_super_token(record)  # 把super_token展开
            record_list.append(record)
        return record_list, args

    @staticmethod
    def split_by_record_def(all_stoken_list, start_idx):
        """
        ... {...}/;
        """
        merge_srecord = []
        for idx in range(start_idx, len(all_stoken_list)):
            token = all_stoken_list[idx]
            merge_srecord.append(token)
            if (isinstance(token, SuperToken) and token[0] == '{' and token[-1] == "}") \
                    or token == ';':
                return merge_srecord, idx + 1
        raise ValueError("not end with ';' or '{...}', but", all_stoken_list[start_idx:start_idx + 10])

    @staticmethod
    def split_by_record_let(all_stoken_list, start_idx):
        """
        xxx in {...}/;
        let a = [] in {
            def A;
            def B;
        }
        let先算作一个大语句
        """
        merge_srecord = []
        for idx in range(start_idx, len(all_stoken_list)):
            token = all_stoken_list[idx]
            merge_srecord.append(token)
            if token == 'in':
                record, end_idx = RecordSplitPass.split_by_record_def(all_stoken_list, idx + 1)
                merge_srecord.extend(record)
                return merge_srecord, end_idx
        raise ValueError("let end error", all_stoken_list[-10:])


class ParserForEachPass(Pass):
    @staticmethod
    def do(srecord, *args):
        """
            foreach xx={xxxx} in DEF
            foreach xx={xxxx} in {A}
        """
        def is_identifier_or_str(x):
            if x.isidentifier():
                return True
            if len(x) >= 2 and x[0] == x[-1] and x[0] in ["'", '"']:
                return True
            return False

        def merge_token(a: Token, b: Token, str_sym={'"': '"', "'": "'"}):
            s, e = '', ''
            if a[0] in str_sym and a[-1] == str_sym[a[0]]:
                s, e, a = a[0], a[-1], a[1:-1]
            if b[0] in str_sym and b[-1] == str_sym[b[0]]:
                b = b[1:-1]
            return Token("{}{}{}{}".format(s, a, b, e))

        def replace_var(sub_record, name, value):
            new_sub_record = []
            i = 0
            while i < len(sub_record):
                if isinstance(sub_record[i], SuperToken):
                    new_sub_record.append(replace_var(sub_record[i], name, value))
                    i += 1
                elif sub_record[i] == '#' and sub_record[i + 1] == name:  # #varname => value
                    """
                    foreach I = 0-31 in def COP0#I : MipsReg<#I, ""#I>;
                    def uimm # I : Operand<i32> {let PrintMethod = "printUImm<" # I # ">";...
                    """
                    if is_identifier_or_str(new_sub_record[-1]):  # merge before token
                        new_sub_record[-1] = merge_token(new_sub_record[-1], value)
                    else:
                        new_sub_record.append(value)
                    i += 2
                    if i + 1 < len(sub_record) and sub_record[i] == '#':
                        new_sub_record[-1] = merge_token(new_sub_record[-1], sub_record[i + 1])
                        i += 2
                else:
                    new_sub_record.append(sub_record[i])
                    i += 1
            return SuperToken(new_sub_record) if isinstance(sub_record, SuperToken) else new_sub_record

        assert srecord.type == 'foreach'
        records_dict = defaultdict(list)  # {'multiclass': [], 'defm': [], 'simple': []}
        in_idx = srecord.index('in')
        if in_idx >= 0:
            iter_stmt = srecord[:in_idx]  # 记录foreach I={..}这个限制
            if len(iter_stmt) == 6:  # foreach var = 1-5
                assert iter_stmt[4] == '-', iter_stmt
                iter_values = [str(i) for i in range(int(iter_stmt[3]), int(iter_stmt[5]) + 1)]
            elif len(iter_stmt) == 4:  # foreach var = {xxx}
                iter_values = iter_stmt[-1][1:-1:2]
            else:
                raise ValueError(srecord)
            var_name = iter_stmt[1]

            for var_value in iter_values:  # iter_stmt[-1]=>{1, 2, 3 ...}, iter_stmt[-1][1:-1:2]=>1 2 3 ...
                token_list = replace_var(srecord[in_idx + 1:], var_name, var_value)
                if isinstance(token_list[0], SuperToken):  # 后面有多条def语句
                    token_list = token_list[0][1:-1]  # 去掉{}
                else:
                    assert token_list[0] in ['def', 'class'], token_list

                srecord.append_attr('constraint', iter_stmt)
                sub_records_dict, _ = ParserInterPass.do(srecord.new(token_list), True, True)
                extend_records_dict(records_dict, sub_records_dict)
        else:
            raise ValueError('unrecognized record', srecord)
        # LOG.debug("(parser_foreach_record)", record)
        # for sub_record in sub_records:
        #     LOG.debug("(parser_foreach_record)", SuperToken.unfold_super_token(sub_record))
        return records_dict, args


class ParserLetPass(Pass):
    @staticmethod
    def do(srecord, *args):
        """
            let xx in DEF
            let xx in { A }
        """
        def add_sep(sub_record):
            final_idx = find_end_token(sub_record)
            # is not end with {...} or ; then append a ;
            if (not SuperToken.is_super_token(sub_record[final_idx], '{', '}')) and sub_record[final_idx] != ';':
                sub_record.append(Token(';'))

        assert srecord.type == 'let'
        in_idx = srecord.index('in')
        if in_idx >= 0:
            if isinstance(srecord[in_idx + 1], SuperToken):  # 后面有多条def语句
                token_list = srecord.new(srecord[in_idx + 1][1:-1])  # 去掉{}
                add_sep(token_list)  # 最后一个非\n空格的符号如果不是}或; 则在最后添加一个;,使得符合split解析的语法
            else:
                # print(srecord)
                assert srecord[in_idx + 1] in ['def', 'class', 'defm', 'let'], srecord[in_idx + 1]
                token_list = srecord[in_idx + 1:]

            let_constraint = srecord[:in_idx]  # 记录let A=[]这个限制
            srecord.append_attr('constraint', let_constraint)
            records_dict, _ = ParserInterPass.do(srecord.new(token_list), True, True)
        else:
            raise ValueError('unrecognized record', srecord)
        return records_dict, args


class AddClassNamePass(Pass):
    NoNameRecordNum = 0

    @staticmethod
    def do(records_dict: dict, *args):
        for r in records_dict['simple'] + records_dict['defm']:
            if r[1] == ':':  # def : or defm :
                r.insert(1, r[0].new(f"NoName{AddClassNamePass.NoNameRecordNum}"))
                AddClassNamePass.NoNameRecordNum += 1
            elif r[1] == '""':
                r[1] = r[1].new(f"NoName{AddClassNamePass.NoNameRecordNum}")
                AddClassNamePass.NoNameRecordNum += 1
        return records_dict, args

    @staticmethod
    def clear_do(records_dict: dict, *args):
        for j, r in enumerate(records_dict["simple"]):
            for ti, t in enumerate(r):
                idx = t.find('NoName')
                if idx == -1 or idx == 0:
                    continue
                end_idx = idx + len('NoName')
                try:
                    c = int(t[end_idx:])
                except ValueError as e:
                    continue
                assert c < AddClassNamePass.NoNameRecordNum, r
                r[ti] = r[ti].new(r[ti][:idx])
        return records_dict, args


class ParserInterPass(Pass):
    @staticmethod
    def do(token_list: list, is_super_token=False, return_super_token=False, *args):
        """
        DEF: def/class xxx {}/;

        A =
            DEF
            let xx in DEF
            let xx in { A }
            foreach xx={xxxx} in DEF
            foreach xx={xxxx} in {A}

        OUT:
            DEF
            DEF data['let'] = [let xx]
            DEF ....
            DEF
            DEF
        输出全是def形式的record
        """
        record_list, _ = RecordSplitPass.do(token_list, is_super_token=is_super_token,
                                            return_super_token=return_super_token)
        records_dict = defaultdict(list)  # {'multiclass': [], 'defm': [], 'simple': []}
        for record in record_list:
            if record.type in ['def', 'class']:
                records_dict['simple'].append(record)
            elif record.type == 'let':
                sub_records_dict, _ = ParserLetPass.do(record)
                extend_records_dict(records_dict, sub_records_dict)
            elif record.type == 'foreach':
                sub_records_dict, _ = ParserForEachPass.do(record)
                extend_records_dict(records_dict, sub_records_dict)
            elif record.type == 'multiclass':
                records_dict['multiclass'].append(record)
            elif record.type == 'defm':
                records_dict['defm'].append(record)
            elif record.type == 'include':
                pass
            else:
                raise ValueError("unkonwed record type", record.type, record)
        records_dict, _ = AddClassNamePass.do(records_dict, args)
        return records_dict, args


# multi records pass
def diamond_inheritance_check(topology_order_nodes, super2children):
    def get_road(src, dst, before_road, result_road):
        for node in super2children[src]:
            before_road.append(node)
            if node == dst:
                result_road.append(before_road)

    super_map = defaultdict(set)
    for node in topology_order_nodes:
        children = super2children[node]
        for child in children:
            if node in super_map[child]:  # have two road to same super
                print(f"{child} are diamond inherit {node}.")
                return True
            super_map[child].add(node)
    return False


class ParserMultiDefmPass(Pass):
    """
    multiclass / defm
    https://www.jianshu.com/p/5f55d2e9f2bf
    1)
    multiclass A<xxx>: ...{
        def aa: ...
        def bb: ...
    }
    =>
        r1: class Aaa<xxx>: ....
        r2: class Abb<xxx>: ....
        A: [(#NAMEaa, Aaa, r1, has_var_token=False),
            (#NAMEbb, Abb, r2, has_var_token=False)]

    defm B: A
    =>
        def Baa: Aaa ...
        def Bbb: Abb ...

    2)
    multiclass A<T yyy, T zzz>: ...{
        def aa#NAME: ...#NAME...
        def bb#NAME: ...#NAME...
    }
    =>
        VarRecord r1: class Aaa#NAME<T yyy, T zzz>: ...#NAME...
        VarRecord r2: class Abb#NAME<T yyy, T zzz>: ...#NAME...
        // name_of_new_class, name_of_new_super_class, record,
        A: [(aa#NAME, Aaa#NAME, r1, has_var_token=True),
            (bb#NAME, Abb#NAME, r2, has_var_token=True)]

    defm B<T xxx>:A<xxx, 1>;
    =>
        def AaaB<T yyy, T zzz>: ...B... // r1 <= {"NAME": B} if has_var_token
        def aaB<T xxx>:AaaB<xxx, 1>   // aa#NAME<T xxx>: Aaa#NAME<xxx, 1> <= {"NAME": B}
        def AbbB<T yyy, T zzz>: ...B...
        def bbB<T xxx>:bbA<xxx, 1>
    """
    VariableStmt = namedtuple("VariableStmt", ["child_class_name", "super_class_name", "variable_record"])

    @staticmethod
    def do(records_dict, *args):
        multic_stmts, defm_stmts = records_dict['multiclass'], records_dict['defm']
        ordered_multic_name, multic_stmts_data = ParserMultiDefmPass.parser_multiclass1(multic_stmts)
        multic_dict, records = ParserMultiDefmPass.parser_multiclass2(ordered_multic_name, multic_stmts_data)
        records.extend(ParserMultiDefmPass.parser_defm(defm_stmts, multic_dict))
        records_dict['simple'].extend(records)
        return records_dict, args

    @staticmethod
    def parser_multiclass1(multic_stmts):
        G = SimpleGraph()
        stmts_data = {}
        StmtData = namedtuple("StmtData", ["class_name", "args", "supers", "content", "record"])
        # 获得multiclass定义的record名字
        for srecord in multic_stmts:
            class_name, args, supers, content = Record.split_def(srecord)
            assert class_name not in G.get_nodes(), class_name
            stmts_data[class_name] = StmtData(class_name, args, supers, None, srecord)
            G.add_node(class_name)

        # 添加multiclass继承的依赖关系，即给依赖图添加边
        for class_name, data in stmts_data.items():
            supers = data.supers
            for supername, superargs in supers:
                assert supername in stmts_data, supername
                G.add_edge(supername, class_name)

        # 添加multiclass内部defm的依赖关系
        for class_name, data in stmts_data.items():
            srecord = data.record
            if srecord[-1] == ';':  # 没有content部分
                continue
            elif SuperToken.is_super_token(srecord[-1], "{", "}"):  # content
                records_dict, _ = ParserInterPass.do(srecord.new(srecord[-1][1:-1]), is_super_token=True,
                                                     return_super_token=True)
                assert len(records_dict['multiclass']) == 0, records_dict.keys()
                sub_defm_stmts = records_dict['defm']
                for defm_stmt in sub_defm_stmts:
                    defm_class_name, defm_args, defm_supers, defm_content = Record.split_def(defm_stmt)
                    for defm_supername, defm_superargs in defm_supers:
                        if defm_supername in stmts_data:  # if super is multiclass
                            G.add_edge(defm_supername, class_name)
            else:
                raise ValueError(srecord[-1] + " is not ; or {...}")

        nodes = topology_sort(G.get_nodes(), G.get_src2dsts())
        assert not diamond_inheritance_check(nodes, G.get_src2dsts()), "exist diamond_inheritance"
        return nodes, stmts_data

    @staticmethod
    def parser_multiclass2(ordered_multic_name, multic_stmts_data):
        records = []
        var_stmts_dict = defaultdict(list)
        for multic_name in ordered_multic_name:
            data = multic_stmts_data[multic_name]
            srecord, args, supers = data.record, data.args, data.supers
            # to record of var token
            srecord, count = VarToken.match(srecord, ('NAME',))
            if SuperToken.is_super_token(srecord[-1], "{", "}"):  # content
                records_dict, _ = ParserInterPass.do(srecord.new(srecord[-1][1:-1]), is_super_token=True,
                                                     return_super_token=True)
                new_sub_records, var_stmts = ParserMultiDefmPass.parser_multiclass2_def(multic_name, args,
                                                                                          records_dict['simple'])
                var_stmts_dict[multic_name].extend(var_stmts)
                records.extend(new_sub_records)

                ParserMultiDefmPass.parser_multiclass2_defm(records_dict['defm'])

            var_stmts = ParserMultiDefmPass.parser_multiclass2_inherit(multic_name, args, supers, var_stmts_dict)
            var_stmts_dict[multic_name].extend(var_stmts)
        return var_stmts_dict, records

    @staticmethod
    def parser_multiclass2_def(classname, args, content_def_stmts):
        """
        把multiclass里的classname和arg抄下来结合到def 的类名里面去
        """
        var_stmts = []
        new_sub_records = []
        for r in content_def_stmts:
            assert r[0] in ['def', 'class'], r
            r[0] = r[0].new("class")
            # 1. 获得child_class_name, aa / aa#NAME => #NAMEaa / aa#NAME
            child_class_name = super_class_name = r[1]  # aa / aa#NAME
            if isinstance(child_class_name, Token):
                child_class_name = VarToken([VarToken.Var('NAME'), child_class_name])  # #NAMEaa / aa#NAME

            # 2. 获得super_class_name
            has_var_token = VarToken.has_var_token(r[2:])
            if has_var_token:
                if isinstance(super_class_name, VarToken):
                    # aa#NAME=>Aaa#NAME(VarToken)
                    r[1] = VarToken([classname, '_'] + super_class_name.fmt)
                else:
                    # aa => A#NAMEaa
                    r[1] = VarToken([classname, VarToken.Var('NAME'), '_', super_class_name])
            else:  # 如果:后面的内容没有#NAME, 则中间父类与defm的子类名无关, 取消中间父类里的#NAME
                # aa(Token), bb(Token)
                if isinstance(super_class_name, VarToken):
                    super_class_name = super_class_name.get_token_with_value({"NAME": ''})  # aa/aa#NAME => aa
                r[1] = Token("{}{}".format(classname, super_class_name))  # Aaa, Abb
                r[1].line_id = super_class_name.line_id
            super_class_name = r[1]

            # 3. build record or var record
            r = r.new(r[:2] + [args] + r[2:])  # 插入参数,形成新的Record
            r.type = r[0]
            # (aa#NAME, Aaa#NAME, r1, has_var_token) / (aa, Aaa)
            var_stmts.append(ParserMultiDefmPass.VariableStmt(child_class_name, super_class_name, r))
            if not has_var_token:
                new_sub_records.append(r)
        return new_sub_records, var_stmts

    @staticmethod
    def parser_multiclass2_inherit(classname, args, supers, inter_stmts_dict):
        """
        multiclass A<T2 op>{
            def aa: SA<op>;
        }
        multiclass B<T2 op2>{
            def aa: SA<op2>;
        }
        multiclass D<T2 op>: A<op>, B<op>
        defm X:D<xxx>
        =>
        defm X:A<xxx>
        defm X:B<XXX>
        """
        def assert_same_args(def_args, super_use_args):
            def_args_name = tuple(tuple(def_arg[-2]) for def_arg in def_args)
            super_use_args_name = tuple(tuple(use_arg) for use_arg in super_use_args)
            assert def_args_name == super_use_args_name, f"{def_args_name} vs {super_use_args_name}"

        args, _, _ = RecordParser._parser_args_or_content(args, ',')
        supers = RecordParser._parser_supers(supers)

        var_stmts = []
        for supername, superargs in supers:
            assert_same_args(args, superargs)
            var_stmts.extend(inter_stmts_dict[supername])
        return var_stmts

    @staticmethod
    def parser_multiclass2_defm(content_defm_stmts):
        pass

    @staticmethod
    def parser_defm(defm_stmts, multic_dict):
        records = []
        for srecord in defm_stmts:
            LOG.fprint(Flag.parser_record, srecord)
            class_name, args, supers, content = Record.split_def(srecord)
            # assert len(superclasses) == 1, "get multi super class for defm in {}".format(defm_stmt)
            sub_records, idx = [], 0
            super_name, super_args = supers[idx]
            kv = {"NAME": class_name}
            for child_class_name, super_class_name, r in multic_dict[super_name]:
                assert isinstance(child_class_name, VarToken)
                child_class_name = child_class_name.get_token_with_value(kv)
                if isinstance(super_class_name, VarToken):
                    super_class_name = super_class_name.get_token_with_value(kv)
                    super_r = VarToken.get_record_with_value(r, kv)
                    sub_records.append(super_r)
                supers[idx] = (super_class_name, super_args)
                r = Record.build_def(child_class_name, args, supers, content, srecord, class_name.line_id)
                sub_records.append(r)
                # print(defm_stmt)
                # print(len(sub_records), sub_records)
            # if srecord[1] == 'VWADDU_V':
            #     print(srecord)
            #     print(sub_records)
            records.extend(sub_records)
        return records


class LetConstraintPass(Pass):
    @staticmethod
    def do(records_dict: dict, *args):
        for r in records_dict['simple']:
            LetConstraintPass.add_let_constraint_to_content(r)
        return records_dict, args

    @staticmethod
    def add_let_constraint_to_content(record):
        if 'constraint' not in record.data: return
        consts = []
        record.data['constraint'] = []
        for let_consts in record.data['constraint']:
            if let_consts[0] != 'let': continue
            last_i = 1
            for i, c in enumerate(let_consts):
                if c == ',':
                    consts.extend([Token('let')] + let_consts[last_i: i] + [Token(';')])
                    last_i = i + 1
            consts.extend([Token('let')] + let_consts[last_i:] + [Token(';')])
        if len(consts) == 0: return
        # consts = deepcopy(consts)
        final_idx = find_end_token(record)
        if record[final_idx] == ';':
            record[final_idx] = SuperToken(["{", "}"])
        record[final_idx] = SuperToken(record[final_idx][:-1] + consts + [Token("}")])


class RemoveDuplicatePass(Pass):
    @staticmethod
    def do(input_records_set: (list, dict), not_allow_repeat=False):
        record_loc_str = lambda x: "{}:{} {}".format(x.filename, x[0].line_id, x)

        if isinstance(input_records_set, list):
            records_set = {"": input_records_set}
        elif isinstance(input_records_set, dict):
            records_set = input_records_set
        else:
            raise TypeError(f"{type(input_records_set)}")

        non_repeat_records_set = defaultdict(list)
        for k, records in records_set.items():

            non_repeat_records = []
            record_map = {}
            for r in records:
                r_name = r[1]  # r.data["name"]
                assert r.type in ["def", "class", "defm", "multiclass"], r.type
                if r_name in record_map:
                    # print(record_map[r_name])
                    # assert False, r
                    LOG.info("repeat record, ignored:")
                    print(record_loc_str(r))
                    LOG.info('repeat record has already found here:')
                    print(record_loc_str(record_map[r_name]))
                    if not_allow_repeat:
                        assert False, "repeat record"
                else:
                    record_map[r_name] = r
                    non_repeat_records.append(r)

            non_repeat_records_set[k] = non_repeat_records
        if isinstance(input_records_set, list):
            non_repeat_records_set = non_repeat_records_set[""]
        return non_repeat_records_set, ()


class PostParserPass(Pass):
    @staticmethod
    def do(records_dict: dict, *args):
        # 2. (2) unfold_record (3) filter space \r \n \t (4) copy tokens in record
        final_simple_records = []
        for j, r in enumerate(records_dict["simple"]):
            assert r[0] in ['def', 'class'], "{}, {}".format(r, r.type)
            r = SuperToken.unfold_super_token(r)
            final_simple_records.append(r.new([t for t in r if not t.isspace()]))  # filter \n \r space \t

        records_dict["simple"] = [deepcopy(r) for r in final_simple_records]
        return records_dict, args


# ##
class RecordParser(object):
    @staticmethod
    def do_parser_and_set(record, is_super_token=False):
        """
        name: ADD
        args: [(type(list), name(list), value(list)), ....]
        supers: [(super_name, [arg1(list), arg2(list), ...]),
                ...]
        content: [(type(list), name(list), value(list)), ....]
        """
        if is_super_token:
            record = SuperToken.unfold_super_token(record)
        for i, token in enumerate(record):
            token.ridx = i
        srecord = SuperToken.collect_super_token(record)

        class_name, args, supers, content = Record.split_def(srecord)
        RecordParser._set_token_pos(class_name, args, supers, content)

        args, _, _ = RecordParser._parser_args_or_content(args, ',')
        supers = RecordParser._parser_supers(supers)
        content, let_list, stmt_ridx = RecordParser._parser_args_or_content(content, ';', with_ridx=True)

        record.data.update({
            'name': class_name,
            'args': args,
            'supers': supers,
            'content': content,
            "content_let": let_list,
            'content_stmt_ridx': stmt_ridx
        })

    @staticmethod
    def build_record(r):
        def build_args(args):
            if len(args) == 0:
                return []
            tmp = []
            for arg_type, arg_name, arg_value in args:
                tmp = [arg_type, arg_name]
                if arg_value is not None:
                    tmp += ["=", arg_value]
                tmp.append(",")
            return ["<"] + tmp[:-1] + [">"]

        def build_supers(supers):
            if len(supers) == 0:
                return []
            tmp = []
            for super_name, super_args in supers:
                tmp.append(super_name)
                if len(super_args) > 0:
                    tmp.extend(["<"] + super_args + [">"])
                tmp.append(",")
            return tmp[:-1]

        def build_content(content, content_let):
            tmp = ["{"]
            for (c_type, c_name, c_value), let in zip(content, content_let):
                if let:
                    tmp.append("let")
                tmp.append([c_type, c_name, '=', c_value, ';'])
            tmp.append("}")
            return tmp

        assert r[0] in ['def', 'class']
        new_r = [r[0], r.data["name"], ] + build_args(r.data["args"]) + \
            build_supers(r.data["super"]) +build_content(r.data["content"], r.data["content_let"])
        new_r = Record(new_r)
        return new_r

    @staticmethod
    def split_by_token(token_list, sep_token, with_idx=False):
        # 第一步，按逗号分隔开
        arg_list = []
        idx_list = [-1]
        arg = []
        for i, token in enumerate(token_list):
            if token == sep_token:
                arg_list.append(arg)
                idx_list.append(i)
                arg = []
            else:
                arg.append(token)
        arg_list.append(arg)
        if len(arg) > 0:
            idx_list.append(len(token_list))

        if with_idx:
            return arg_list, [(idx_list[i]+1, idx_list[i+1]) for i in range(len(idx_list)-1)]
        return arg_list

    @staticmethod
    def _parser_args_or_content(args, sep, with_ridx=False):
        """
        T[<...>] var=value [,...]
        """
        if args is None:
            return [], [], []
        args = args[1:-1]
        # (type, var, value)
        arg_list = []
        let_list = []
        ridx_list = []
        for arg, (start_i, end_i) in zip(*RecordParser.split_by_token(args, sep, True)):
            if len(arg) == 0: continue
            arg = RecordParser.split_by_token(arg, '=')
            assert len(arg) < 3, arg
            if len(arg) == 2:
                value = arg[1]
            else:
                value = None
            arg = arg[0]
            var = arg[-1]
            let = False
            if isinstance(var, SuperToken):
                var = arg[-2:]
            elif isinstance(var, Token):
                var = arg[-1:]
            if len(arg) > len(var):
                type = arg[:-len(var)]
                if type[0] == 'let':
                    type = type[1:]
                    let = True
            else:
                type = None
            arg_list.append((type, var, value))
            let_list.append(let)
            if with_ridx:
                ridx_list.append((get_first_token(args[start_i]).ridx, get_last_token(args[end_i]).ridx+1))

        return arg_list, let_list, ridx_list

    @staticmethod
    def _parser_supers(supers_data):
        supers = []
        for super_name, super_args in supers_data:
            if super_args is not None:
                # arg can be => !strconcat(opstr , "\t$fd, $fr, $fs, $ft")
                super_args = [t for t in RecordParser.split_by_token(super_args[1:-1], ',')]  # results of split is list[list]
            else:
                super_args = []
            supers.append((super_name, super_args))
        return supers

    @staticmethod
    def _set_token_pos(class_name, args, supers, content):
        def set_pos(super_token, pos):
            if super_token is not None:
                for t in SuperToken.unfold_super_token(super_token):
                    t.pos = pos

        class_name.pos = TokenPos.CLASSNAME
        set_pos(args, TokenPos.ARGS)
        for super_name, super_args in supers:
            super_name.pos = TokenPos.CLASSNAME
            set_pos(super_args, TokenPos.SUPER_ARGS)
        set_pos(content, TokenPos.CONTENT)
