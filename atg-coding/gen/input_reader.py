import csv


def csv_to_dict(filename):
    row_data = []
    f = open(filename)
    reader = csv.reader(f)
    for row in reader:
        row_data.append(row)
    f.close()

    data = {}
    attrs_name = row_data[0]    # 第一行是属性名
    for d in row_data[1:]:
        single_data = {}
        for i, attr_name in enumerate(attrs_name):
            single_data[attr_name] = d[i].strip()

        for key in single_data:
            if key.lower() == 'rname_upper':
                inst_name = single_data[key]
        data[inst_name] = single_data
    return data


def parse_frag_to_dict(value):
    """
    {
      "name": "imm12",
      "parts":[
            {"range": [a, b], "Inst_range": [c, d]},
            {"range": [a, b], "Inst_range": [c, d]}
        ],
      "value": xxx
      "is_op": xxx
     ...
    }
    """
    def str_to_bool(s):
        if s.lower() == 'true':
            return True
        elif s.lower() == 'false':
            return False
        else:
            raise ValueError(f"{s}")

    def str_to_int_tuple(x):
        a, b = x.split('-')
        return int(a), int(b)

    def cal_frag_length(frag):
        l = 0
        for part in frag['parts']:
            inst_e, inst_s = part['inst_range']
            l += abs(inst_e - inst_s) + 1
        return l

    def full_binary(s):
        for e in s:
            if e not in ['0', '1']:
                return False
        return True

    res = {"parts": []}
    value = value.split(';')
    value[0] = value[0][1:]     # remove {
    value[-1] = value[-1][:-1]  # remove }
    for v in value:
        v = v.strip()
        if v.startswith('{') and v.endswith('}'):
            vs = v[1:-1].split("#")
            frag = {}
            for v in vs:
                k, v = v.split('=')
                frag[k] = v
            res['parts'].append(frag)
        else:
            k, v = v.split('=')
            res[k] = v

    res['is_split'] = str_to_bool(res['is_split'])
    if not res['is_split']:
        assert len(res['parts']) == 0
        part = {'bit_range': res['bit_range']}
        if 'frag_range' in res:
            part['frag_range'] = res['frag_range']
        res['parts'].append(part)
        del res['bit_range']

    for part in res['parts']:
        part['inst_range'] = str_to_int_tuple(part['bit_range'])
        del part['bit_range']
        if 'frag_range' in part:
            part['range'] = str_to_int_tuple(part['frag_range'])
            del part['frag_range']
            if 'frag_id' in part:
                del part['frag_id']
    del res['is_split']

    res['value'] = res['value'].strip()
    frag_len = cal_frag_length(res)
    if res['value'].startswith('0b'):
        assert full_binary(res['value'][2:])
        res['value'] = "0b" + res['value'][2:].zfill(frag_len)
    elif res['value'].isnumeric():
        if frag_len == len(res['value']):  # binary
            assert full_binary(res['value'])
            res['value'] = f'0b{res["value"]}'
        else:  # demical => binray
            if len(res['value']) > 1 and full_binary(res['value']):
                raise ValueError(f"{res['value']} not support for length {frag_len}")
            else:  # demical => binray
                value = bin(int(res['value']))
                res['value'] = "0b" + value[2:].zfill(frag_len)
    else:
        assert res['value'] == 'NULL', res['value']

    # print(value)
    # x = json.loads(value)
    # print(x)
    return res


def map_value(data):
    def map_str_to_value(v):
        v = v.strip()
        if v.lower() in ['none', 'null', '?']:
            return None
        elif v.lower() in ['true', 'yes']:
            return True
        elif v.lower() in ['false', 'no']:
            return False
        elif v.isnumeric():
            return v
        else:
            return v

    for k, v in data.items():
        if isinstance(v, dict):
            map_value(v)
        if isinstance(v, list):
            for e in v:
                assert isinstance(e, dict)
                map_value(e)
        elif isinstance(v, str):
            data[k] = map_str_to_value(v)


def parse_json_value(data):
    for inst_name, inst_attrs in data.items():
        # 1. -> complete dict
        for attr_name, attr_value in inst_attrs.items():
            attr_value = attr_value.strip()
            if attr_value.startswith('{') and attr_value.endswith('}'):   # is a json dict format
                inst_attrs[attr_name] = parse_frag_to_dict(attr_value)

        # 2. -> str to value
        map_value(inst_attrs)

        # 3. frag1, frag2, ... => frags: [frag1, frag2, ...]
        frags = []
        for attr_name in list(inst_attrs.keys()):
            if attr_name.startswith('frag'):
                # print(attr_name, inst_attrs[attr_name])
                if inst_attrs[attr_name] is not None:
                    frags.append(inst_attrs[attr_name].copy())
                del inst_attrs[attr_name]
        inst_attrs['frags'] = frags
        # print(inst_attrs)


def read_input_data(filename):
    data = csv_to_dict(filename)
    parse_json_value(data)
    return data


if __name__ == '__main__':
    data = csv_to_dict('Input/inst_set_input/I/I.csv')
    parse_json_value(data)
    print(data['beq'])

