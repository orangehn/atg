from enum import Enum
from copy import deepcopy


class TokenColor(Enum):
    NOCOLOR = 0
    BLUE = 1
    GREEN = 2
    RED = 3
    PINK = 4


class TokenPos(Enum):
    CLASSNAME = 0  # class name or super class name
    ARGS = 1
    SUPER_ARGS = 2
    CONTENT = 3


class Location(object):
    """
    记录一个token当前出现的位置信息：
    文件名
    哪个记录
    第几个token
    length
    """
    instance = {}
    check_ = False  # 加下划线的变量类似于私有变量

    @staticmethod
    def create_location(filename, record_id, token_id, length=1, line_id=None):
        # 常数池，为的是解决多次创建同一个位置，不同对象指针不同使得等号失效
        key = (filename, record_id, token_id, length)
        if key not in Location.instance:
            Location.check_ = True
            loc = Location.instance[key] = Location(filename, record_id, token_id, length, line_id)
            Location.check_ = False
        else:
            loc = Location.instance[key]
        assert line_id is None or loc.line_id == line_id, \
            'given line_id must be same as before define {} vs {}'.format(loc.line_id, line_id)
        if line_id is None and length > 1:  # supertoken的第一个token的line_id
            loc1 = Location.instance[(filename, record_id, token_id, 1)]
            loc.line_id = loc1.line_id
        return loc

    def __init__(self, filename, record_id, token_id, length=1, line_id=None):
        assert Location.check_, "constructor must be involved by Location.create_location(...)"
        self.filename = filename
        self.record_id = record_id
        self.token_id = token_id
        self.length = length
        self.line_id = line_id

    def __str__(self):
        return f"{self.filename}:{self.line_id} rid={self.record_id},tid={self.token_id},len={self.length}"  # 类似c++中的占位符: %s %d

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        if self.filename < other.filename:
            return True
        elif self.filename == other.filename:
            if self.record_id < other.record_id:
                return True
            elif self.record_id == other.record_id:
                if self.token_id < other.token_id:
                    return True
                elif self.token_id == other.token_id:
                    if self.length < other.length:
                        return True
        return False


class Token(str):
    """
    class继承自str, str所有方法都可以直接拿过来用
    """

    def __init__(self, token_str, color=TokenColor.NOCOLOR):
        super(Token, self).__init__()
        self.line_id = None  # for debug, line id of token in origin file
        self.loc = None  # token location
        self.local = False  # global or local means if token in a scope

        self.color = color  # for Agile Syntax
        self.du = None  # token is DEF or USE， None, "U", "D" , initial_value
        self.to = None  # data flow from here to one location

        self.pos = None  # TokenPos

    def new(self, token_str):
        t = Token(token_str)
        t.line_id = self.line_id
        t.loc = self.loc
        t.local = self.local
        t.color = self.color
        t.du = self.du
        t.to = self.to
        return t


class SuperToken(list):
    close_items = {'{': '}', '[': ']', '<': '>', '(': ')'}

    def __init__(self, tokens):
        super(SuperToken, self).__init__(tokens)
        self.du = None
        self.color = TokenColor.NOCOLOR

    def new(self, tokens):
        st = SuperToken(tokens)
        st.du = self.du
        st.color = self.color
        return st

    @staticmethod
    def collect_super_token(token_list, close_items=None, start_id=0, end_item=None, recursion=True):
        r, _ = SuperToken._collect_super_token(token_list, close_items, start_id, end_item, recursion)
        if isinstance(token_list, Record):
            r = Record(r)
            r.set_attr_as(token_list)
        elif isinstance(token_list, list):
            pass
        else:
            raise TypeError()
        return r

    @staticmethod
    def unfold_super_token(stoken_list):
        r = SuperToken._unfold_super_token(stoken_list)
        if isinstance(stoken_list, Record):
            r = Record(r)
            r.set_attr_as(stoken_list)
        elif isinstance(stoken_list, list):
            pass
        else:
            raise TypeError()
        return r

    @staticmethod
    def _collect_super_token(token_list: list, close_items=None, start_id=0, end_item=None, recursion=True):
        def find_end(token_list, si, end_items):
            l = len(end_items)
            for i in range(si, len(token_list) - l):
                find = True
                for j in range(l):
                    if token_list[i + j] != end_items[j]:
                        find = False
                if find:
                    return token_list[i:i + l], i + l
            raise ValueError("can not find end with ({}) {}".format(end_items, token_list))

        if close_items is None:
            close_items = SuperToken.close_items

        # 递归合并SuperToken
        i = start_id
        collected_tokens = []
        find = False
        while i < len(token_list):
            token = token_list[i]
            if token == end_item:
                find = True
                break
            elif recursion and token in close_items:  # [{}] 表示代码段
                if token == '[' and token_list[i + 1] == '{':  # to avoid < or > in code segment
                    _, end_i = find_end(token_list, i + 2, "}]")
                    super_token = token_list[i:end_i]
                    collected_tokens.append(SuperToken(super_token))
                    i = end_i
                else:
                    # old_i = i
                    # print("left", old_i, token_list[i], token_list[i-4:i+4])
                    stokens, i = SuperToken._collect_super_token(token_list, close_items, i + 1, close_items[token])
                    # print("right", old_i, token_list[i], token_list[i-4:i+4])
                    super_token = SuperToken([token] + stokens + [token_list[i]])
                    collected_tokens.append(super_token)
                    i += 1
            else:
                collected_tokens.append(token)
                i += 1
        assert end_item is None or find, "{} not in {}".format(end_item, token_list[start_id:])
        return collected_tokens, i

    @staticmethod
    def _unfold_super_token(stoken_list: list):
        res = []
        for token in stoken_list:
            if isinstance(token, SuperToken):
                res.extend(SuperToken._unfold_super_token(token))
            else:
                res.append(token)
        return res

    @staticmethod
    def is_super_token(t, l, r):
        return isinstance(t, SuperToken) and t[0] == l and t[-1] == r


class Record(list):
    """
    single stmt
    """
    def __init__(self, token_strs, type=None, origin=None):
        """
        origin: 记录对应的原本的形式的record
        """
        # token_str是否是Token类的对象，如果不是，将它转成Token类型的
        lines = []
        for token_str in token_strs:
            if isinstance(token_str, (Token, SuperToken, VarToken)):
                lines.append(token_str)
            elif isinstance(token_str, str):
                lines.append(Token(token_str))
            else:
                raise ValueError(f"{token_str}")
        super(Record, self).__init__(lines)
        self.type = type
        self.origin = origin
        self.data = {}  # for def/class, name, argv, data["constraint"] = [[let x=1], foreach]
        self.filename = None

    def append_attr(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def set_attr_as(self, r):
        self.type = r.type
        self.data = deepcopy(r.data)
        self.origin = r.origin
        self.filename = r.filename

    def new(self, tokens):
        r = Record(tokens)
        r.set_attr_as(self)
        return r

    def print(self, with_du=False):
        if with_du:
            print([f"{x}({x.du})" for x in self])
        else:
            print(self)
        if hasattr(self[0], 'loc'):
            print(self[0].loc)
        print("data: {")
        for k, v in self.data.items():
            print("\t", k, ":", v)
        print("}")

    # for super_token Record
    @staticmethod
    def split_def(srecord):
        """
        input: supertoken record
        class_name args: (superclasses, arg)*
        """
        r = srecord
        assert r[0] in ['class', 'def', 'defm', 'multiclass']
        assert r[1] != ':', srecord

        class_name = r[1]
        i = 2
        if SuperToken.is_super_token(r[i], '<', '>'):
            args = r[i]
            i += 1
        else:
            args = None
        supers = []
        if r[i] == ':':
            i += 1
            while i < len(r) - 1:
                super_name = r[i]
                i += 1
                if SuperToken.is_super_token(r[i], '<', '>'):
                    super_args = r[i]
                    i += 1
                else:
                    super_args = None
                if i < len(r) - 1:
                    assert r[i] == ',', srecord
                    i += 1
                supers.append((super_name, super_args))
        assert i == len(r)-1, srecord
        if SuperToken.is_super_token(r[i], '{', '}'):
            content = r[-1]
        else:
            content = None
        return class_name, args, supers, content

    @staticmethod
    def build_def(class_name, args, supers, content, origin_r, line_id=None):
        """
        class_name: Token
        args: SuperToken or None
        supers: [(name, SuperToken or None) ...]
        content: ; or SuperToken({...})
        """
        def format_list(name, args):
            return [name] + ([] if args is None else [args])
        r = ['def'] + format_list(class_name, args) + [':']
        for super_name, super_args in supers:
            r.extend(format_list(super_name, super_args) + [','])
        r[-1] = ';' if content is None else content
        nr = []
        for t in r:
            if isinstance(t, str):
                t = Token(t)
                t.line_id = line_id
            elif isinstance(t, (Token, SuperToken)):
                pass
            else:
                raise TypeError(f"{type(t)}")
            nr.append(t)

        r = origin_r.new(nr)
        r.type = r[0]
        return r


def to_records(record_list):
    return [Record(record) for record in record_list]


def fmt_print(tokens: list, indent=''):  # indent: 缩进
    if isinstance(tokens[0], str):
        # ["class", "ISA_MIPS1", "{"]
        return indent + " ".join(tokens)  # 使用空格将tokens连接起来
    elif isinstance(tokens[0], list):
        # [["class", "ISA_MIPS1", "{"], ... ["def", "A", ";"]]
        return "\n".join([indent + fmt_print(token, indent + ' ') for token in tokens])


class VarToken(object):
    class Var(object):
        def __init__(self, name_token):
            self.name_token = name_token

    def __repr__(self):
        return "{}".format("".join("#{}".format(t.name_token) if isinstance(t, VarToken.Var) else t for t in self.fmt))

    def __str__(self):
        return self.__repr__()

    def __init__(self, fmt):
        self.fmt = fmt
        self.var_idx = {}
        self.quote = None
        for i, x in enumerate(self.fmt):
            if isinstance(x, VarToken.Var):
                self.var_idx[x.name_token] = i
            elif isinstance(x, (Token, str)):
                if x[0] == x[-1] and x[0] in ['"', "'"]:
                    self.quote = x[0]
                    self.fmt[i] = x[1:-1]
            else:
                raise TypeError()

    def get_token_with_value(self, key_value):
        fmt = self.fmt.copy()
        for key, value in key_value.items():
            idx = self.var_idx[key]
            fmt[idx] = value
        string = "".join(fmt)
        if self.quote is not None:
            string = "{}{}{}".format(self.quote, string, self.quote)
        token = Token(string)
        first_token = self._get_first_token()
        token.line_id = first_token.line_id
        return token

    def _get_first_token(self):
        first_token = None
        for t in self.fmt:
            if isinstance(t, Token):
                first_token = t
            elif isinstance(t, VarToken.Var) and isinstance(t.name_token, Token):
                first_token = t.name_token
        return first_token

    @staticmethod
    def _replace_to_var(record, names):
        new_record = []
        count = {name: 0 for name in names}
        for t in record:
            find = False
            for name in names:
                if t == name:
                    new_record.append(VarToken.Var(t))
                    find = True
                    count[name] += 1
                    break
            if not find:
                new_record.append(t)
        return new_record, count

    @staticmethod
    def _merge_by_sharp(token_list):
        i = 0
        new_reocrd = []
        while i < len(token_list)-1:
            if token_list[i+1] == '#':
                fmt = [token_list[i], token_list[i+2]]
                i += 3
                while i < len(token_list)-1 and token_list[i] == '#':
                    fmt.append(token_list[i+1])
                    i += 2
                new_reocrd.append(VarToken(fmt))
            if i < len(token_list)-1:
                new_reocrd.append(token_list[i])
                i += 1
        new_reocrd.append(token_list[-1])

        for i, t in enumerate(new_reocrd):
            if isinstance(t, VarToken.Var):
                new_reocrd[i] = VarToken([t])  # single NAME with out #
        return new_reocrd

    @staticmethod
    def match(srecord, names):
        record = SuperToken.unfold_super_token(srecord)
        token_list, count = VarToken._replace_to_var(record, names)
        var_token_list = VarToken._merge_by_sharp(token_list)
        var_record = record.new(var_token_list)

        var_srecord = SuperToken.collect_super_token(var_record)
        return var_srecord, count

    @staticmethod
    def has_var_token(srecord):
        for t in SuperToken.unfold_super_token(srecord):
            if isinstance(t, VarToken):
                return True
        return False

    @staticmethod
    def get_record_with_value(var_srecord, kv:dict):
        r = []
        for t in var_srecord:
            if isinstance(t, SuperToken):
                t = VarToken.get_record_with_value(t, kv)
            elif isinstance(t, VarToken):
                t = t.get_token_with_value(kv)
            r.append(t)
        if isinstance(var_srecord, (Record, SuperToken)):
            r = var_srecord.new(r)
        return r
