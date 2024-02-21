### important data structure

"""
3.22 update:
 1. record 添加 line_id
"""

"""
1. def/class 定义inst, 可以被分为
def HasV8_1MMainlineOps : SubtargetFeature<
               "v8.1m.main", "HasV8_1MMainlineOps", "true",
               "Support ARM v8-1M Mainline instructions",
               [HasV8MMainlineOps]>;

let constraint
start_keyword
inst_name
superclasses
superclass_args
content

2. foreach xxx in inst
foreach i = {1-7,9-15,18,20-28} in
    def FeatureReserveX#i : SubtargetFeature<"reserve-x"#i, "ReserveXRegister["#i#"]", "true",
                                             "Reserve X"#i#", making it unavailable "
                                             "as a GPR">;
let xxx in inst


对于关键字let
"""

"""
1. 读入数据，删除注释，词法分析，合并引号内的内容为一个token, 记录每个token的行号token.line_id
    //, /**/
    "// Pseudo use of $src", all-target/NVPTX/NVPTXInstrInfo.td:2253
    [/*(set GPR32:$Rd, (AccNode GPR32:$Ra, (mul GPR32:$Rn, GPR32:$Rm)))*/]>, all-target/AArch64/AArch64InstrFormats.td:1858
    defset, ./all-target/NVPTX/NVPTXIntrinsics.td:7573
2. 把token_list转换成supertoken, 按关键字分割出一条条record, 将supertoken展开成一般的token list
split_record:
def/class 直接记录下来

3. 词法分析
语法分析
  分解语句
  按类型处理语句
     def/class/defm/multi class直接记录
     let 把前缀记下来，后面递归调用处理
    foreach 变量展开调用，对每条的后面递归调用
  处理multi class和defm
  把let约束添加到body里
"""
from parser_util_v2.dataset_type import *
from parser_util_v2.dataset_parser import *
from parser_util_v2.project_utils import LOG
from copy import deepcopy
from collections import defaultdict
import os
from tqdm import tqdm


class Dataset(object):
    log_level = LOG.DEBUG

    def __init__(self, dir):
        # 每个文件与其记录做一个映射，形成一个字典； 或者文件名与其记录形成一个tuple，所有文件构成一个list
        self.documents_dict = Dataset.get_records(dir)
        self.documents_list = list(self.documents_dict.items())
        self.set_token_location()
        self.set_token_local()

        self.records_map = {}
        self.records_fig = {}
        # parser for record
        for filename, records in self:
            for record in records:
                assert record.data['name'] not in self.records_map
                self.records_map[record.data['name']] = record
                self.records_fig[record.data["name"]] = len(self.records_fig)

    @staticmethod
    def get_records(dir):
        """
        静态函数，由staticmethod定义
        加上该限制，类Dataset不需要实例化便可以直接调用该函数
        """
        filenames = []
        if os.path.isdir(dir):
            for home, dirs, files in os.walk(dir):
                for file in files:
                    fullname = os.path.join(home, file)
                    if fullname.endswith('.td'):
                        filenames.append(fullname)
        else:
            # 传入的不是目录，直接是文件名
            filenames.append(dir)

        documents = get_dataset(filenames)
        return documents

    def __getitem__(self, idx):  # 定义中括号获取，同时使得对象可以迭代，主要定义对对象操作，返回啥
        if isinstance(idx, int):
            return self.documents_list[idx]
        elif isinstance(idx, str):
            return self.documents_dict[idx]
        elif isinstance(idx, Location):
            return self.documents_dict[idx.filename][idx.record_id]
        else:
            raise ValueError(f"only support int or str as index, but got {idx}")  # 手动引发异常

    def get_document(self, loc: Location):
        return self.documents_dict[loc.filename]

    def get_record(self, loc: Location):
        return self.documents_dict[loc.filename][loc.record_id]

    def set_token_location(self):
        for (filename, records) in self:
            for record_id, record in enumerate(records):
                for j, token in enumerate(record):
                    token.loc = Location.create_location(filename, record_id, j, line_id=token.line_id)

    def set_token_local(self):
        def set_local(super_token):
            for token in super_token:
                if isinstance(token, SuperToken):
                    set_local(token)
                else:
                    token.local = True

        for (filename, records) in self:
            for record_id, record in enumerate(records):
                record = SuperToken.collect_super_token(record, SuperToken.close_items)
                for token in record:
                    if isinstance(token, SuperToken):
                        set_local(token)

    def get_tokens_by_loc(self, loc):
        """
        通过location 获得token str
        """
        return self[loc.filename][loc.record_id][loc.token_id:loc.token_id+loc.length]

    def get_record_fig(self, loc):
        src_record = self.get_record(loc)
        return self.records_fig[src_record.data["name"]]

    def get_record_loc(self, record_name):
        return self.records_map[record_name][1].loc

    def print(self):
        for filename, records in self:
            print(filename)
            for record in records:
                print("\t", record)


# ## function for dataset constructor


def readlines(filename, encodings=['utf-8', 'gbk']):
    """
    只处理utf-8和gbk两种编码
    """
    for encoding in encodings:
        try:
            lines = open(filename, encoding=encoding).readlines()
            return lines
        except UnicodeDecodeError as e:
            pass  # 其他编码，不处理
    return None


def read(filename, encodings=['utf-8', 'gbk']):
    """
    只处理utf-8和gbk两种编码
    """
    for encoding in encodings:
        try:
            data = open(filename, encoding=encoding).read()  # 一个文件一个字符串
            return data
        except UnicodeDecodeError as e:
            pass  # 其他编码，不处理
    return None


def parser(origin_records):
    """
    按类型处理语句 parser_r
        def/class/defm/multi class直接记录
        let 把前缀记下来，后面递归调用处理
        foreach 变量展开调用，对每条的后面递归调用
    处理multi class和defm parser_multi
    把let约束添加到body里 add_let_constraint_to_content
    复制每条语句中的Token
    """
    def set_origin_i(rd, i):
        for k, rs in rd.items():
            for r in rs:
                r.origin = i

    records_dict = defaultdict(list)
    # let/foreach转化 + multiclass/defm的收集
    # 1. (1) to super token (2) set origin i (3) collect complex stmts(multiclass, defm) (4) convert to simple records
    for i, ori_record in enumerate(origin_records):
        super_record = SuperToken.collect_super_token(ori_record)
        sub_records_dict, _ = ParserInterPass.do(super_record, is_super_token=True, return_super_token=True)
        set_origin_i(sub_records_dict, i)
        extend_records_dict(records_dict, sub_records_dict)

    # parser need intra-records:  multiclass/defm的收集
    records_dict, _ = ParserMultiDefmPass.do(records_dict)

    # 2. (1) add constraints (2) unfold_record (3) filter space \r \n \t (4) copy tokens in record
    records_dict, _ = LetConstraintPass.do(records_dict)
    records_dict, _ = PostParserPass.do(records_dict)

    # ADDNoName1 --> ADD
    records_dict, _ = AddClassNamePass.clear_do(records_dict)

    # 3. parser record, add "name", "args", "supers", "content" in r.data
    for r in records_dict['simple']:
        RecordParser.do_parser_and_set(r)
    return records_dict['simple']


def lexical_analyser(data):
    """
    stmts:  def ADD:A ;\n def A ;
    line_id:0   0  0 0  0 1   1 1
    """

    def find_end_idx(data, start_idx, key):
        end_idx = data[start_idx:].find(key)
        if end_idx >= 0:
            return start_idx + end_idx + len(key)
        return -1

    idx, token_list, tokens_line_id, collect_token = 0, [], [], []
    cur_line_id = 1  # 为了debug记录的行号
    while idx < len(data):
        c = data[idx]
        # print(idx, c, cur_line_id)
        if c == '"' or c == "'":  # 遇到引号，找到下一个引号，作为一个token
            end_idx = find_end_idx(data, idx + 1, c)
            assert end_idx >= 0
            token = data[idx:end_idx]
            assert '\n' not in token, "\\n in token {}".format(token)  # 默认引号内容不换行
            token_list.append(token)
            tokens_line_id.append(cur_line_id)  # for debug
            idx = end_idx
            continue
        elif c == '/':  # 可能遇到注释
            if data[idx + 1] == '/':  # //注释，跳过一行
                idx = find_end_idx(data, idx + 1, '\n')
                if idx == -1:  # 当前即是最后一行
                    idx = len(data)
                cur_line_id += 1
                continue
            elif data[idx + 1] == "*":  # /*注释，找到下一个*/, 跳过
                end_idx = find_end_idx(data, idx + 1, '*/')
                assert end_idx >= 0
                cur_line_id += data[idx:end_idx].count('\n')  # 可能跨多行
                idx = end_idx
                continue
        if c.isalnum() or c == '_':  # 字母数字下划线，identifier的组成
            collect_token.append(c)
        else:  # 遇到特殊符号，形成一个新的token
            if len(collect_token) > 0:
                token_list.append("".join(collect_token))
                tokens_line_id.append(cur_line_id)  # for debug
                collect_token = []
            if not c.isspace():  # \t\r\n空格都是space
                token_list.append(c)
                tokens_line_id.append(cur_line_id)  # for debug
            if c == '\n':
                token_list.append(c)
                tokens_line_id.append(cur_line_id)  # for debug
                cur_line_id += 1
        idx += 1

    assert len(token_list) == len(tokens_line_id)
    final_token_list = []
    for token, line_id in zip(token_list, tokens_line_id):
        token = Token(token)
        token.line_id = line_id
        final_token_list.append(token)
    return final_token_list


def get_dataset(filenames):
    """
    词法分析
    语法分析
        分解语句
    多文件一起处理
        调用 parser处理不同语句, multiclass defm需要records之间变换，不同records会跨文件
    """
    documents = defaultdict(list)

    # 词法分析 + 语法分析(分解语句)
    records = []
    token_sum = 0
    for file_name in filenames:
        data = read(file_name)
        token_list = lexical_analyser(data)
        # token_list = [t for t in token_list if t != '\n']
        token_sum += len(token_list)
        rs, _ = RecordSplitPass.do(token_list)
        for r in rs:
            r.filename = file_name
        rs = [r.new([t for t in r if t != '\n']) for r in rs]
        records.extend(rs)
    # print("305 atg token sum, before parser",token_sum)

    # # 多文件一起处理 语法分析
    records = parser(records)
    records, _ = RemoveDuplicatePass.do(records, False)   # duplicate check

    # 按文件名分开
    for r in records:
        assert r.filename is not None
        documents[r.filename].append(r)
    return documents


if __name__ == "__main__":
    record_list = get_dataset(["../MipsInstrInfo-all.td"])
    for record in record_list:
        print(record)
    print()
    record = ['class', 'ADD_FM', '<', 'bits', '<', '6', '>', 'op', ',', 'bits', '<', '6', '>', 'funct', '>', ':',
              'StdArch', '{', 'bits', '<', '5', '>', 'rd', ';', 'bits', '<', '5', '>', 'rs', ';', 'bits', '<', '5', '>',
              'rt', ';', 'bits', '<', '32', '>', 'Inst', ';', 'let', 'Inst', '{', '31', '-', '26', '}', '=', 'op', ';',
              'let', 'Inst', '{', '25', '-', '21', '}', '=', 'rs', ';', 'let', 'Inst', '{', '20', '-', '16', '}', '=',
              'rt', ';', 'let', 'Inst', '{', '15', '-', '11', '}', '=', 'rd', ';', 'let', 'Inst', '{', '10', '-', '6',
              '}', '=', '0', ';', 'let', 'Inst', '{', '5', '-', '0', '}', '=', 'funct', ';', '}']
    record, _ = SuperToken.collect_super_token(record)
    print(record)
