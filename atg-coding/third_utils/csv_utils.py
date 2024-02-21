import csv
from copy import deepcopy
from collections import defaultdict
# a += b
# csv.reader('Input/inst_set_input/v/rvv(1).csv')


class CSVObject(object):
    def __init__(self, data, title_to_lower=False):
        if isinstance(data, str):
            self.row_data = self.read(data)  # filename
        elif isinstance(data, list):
            self.row_data = data
        if title_to_lower:
            self.row_data[0] = [d.lower() for d in self.row_data[0]]

    def update_title(self, titles_map):
        for i, old_title in enumerate(self.row_data[0]):
            if old_title in titles_map:
                self.row_data[0][i] = titles_map[old_title]

    def add_columns(self, attrs, no_repeated=False):
        if no_repeated:
            attrs_names = list(set(attrs.keys()) - set(self.row_data[0]))
        else:
            attrs_names = list(attrs.keys())
        attrs_values = [attrs[name] for name in attrs_names]
        self.row_data[0].extend(attrs_names)
        for row in self.row_data[1:]:
            row.extend(attrs_values)

    def read(self, filename):
        row_data = []
        f = open(filename)
        reader = csv.reader(f)
        for row in reader:
            row_data.append(row)
        f.close()
        return row_data

    def get_key2idx(self):
        key2idx = {}
        for i, key in enumerate(self.row_data[0]):
            key2idx[key] = i
        return key2idx

    def to_key_dict(self, key):
        idx = self.get_key2idx()[key]
        data = {}
        for row in self.row_data[1:]:
            assert row[idx] not in data, f"there are multi rows whose {key}={row[idx]}"
            data[row[idx]] = row
        return data

    def to_attr_dict(self, attr):
        idx = self.get_key2idx()[attr]
        data = defaultdict(list)
        for row in self.row_data[1:]:
            assert row[idx] not in data, f"there are multi rows whose {key}={row[idx]}"
            data[row[idx]] = row
        return data

    def save(self, filename, attr_names=None):
        writer = csv.writer(open(filename, 'w', newline=''))  # newline=''不加这个，每条记录后会有多余的空行
        if attr_names is None:
            attr_names = self.row_data[0]
        writer.writerow(attr_names)

        key2idx = self.get_key2idx()
        new_csv_data = []
        for row in self.row_data[1:]:
            data = [row[key2idx[key]] for key in attr_names]
            data = [('None' if d is None else d) for d in data]
            new_csv_data.append(data)
        writer.writerows(new_csv_data)

    def select_columns_by_key(self, keys, ignore_miss=False):
        """
        key
        """
        key2idx = self.get_key2idx()
        if ignore_miss:
            keys = [key for key in keys if key in key2idx]
        new_csv_data = []
        for row in self.row_data[1:]:
            data = [row[key2idx[key]] for key in keys]
            new_csv_data.append(data)
        new_csv_data = [keys] + new_csv_data
        return CSVObject(deepcopy(new_csv_data))

    def select_rows_by_key(self, key, values, ignore_miss=False):
        """
        row[key] in values
        """
        title = self.row_data[0]
        csv_dict = self.to_key_dict(key)

        if not ignore_miss:
            ret_csv_data = [title] + [csv_dict[v] for v in values]
        else:
            ret_csv_data = [title] + [csv_dict[v] for v in values if v in csv_dict]
        return CSVObject(deepcopy(ret_csv_data))

    def select_row_by_attr(self, attr, values, ignore_miss=False):
        key2idx = self.get_key2idx()
        idx = key2idx[attr]

    def __getitem__(self, items):
        if isinstance(items, tuple):
            row_idx, col_idx = items
        else:
            row_idx, col_idx = items, None

        if row_idx is None:                                 # csv_obj[None]
            row_data = self.row_data[1:]
        elif isinstance(row_idx, slice):                    # csv_obj[:]
            assert row_idx.start is None and row_idx.stop is None and row_idx.step is None
            row_data = self.row_data[1:]
        elif isinstance(row_idx, list):                     # csv_obj[['3', '5', '8']]
            csv_dict = self.to_key_dict(self.row_data[0][0])
            row_data = [csv_dict[d] for d in row_idx]
        elif isinstance(row_idx, dict):  # csv_obj[{"age": "3"}], csv_obj[{"age": [3, 5]}]
            assert len(row_idx) == 1
            key, row_idx = list(row_idx.items())[0]
            csv_dict = self.to_key_dict(key)
            if isinstance(row_idx, str):                   # csv_obj[{"age": "3"}]
                row_data = csv_dict[row_idx]
            elif isinstance(row_idx, (tuple, list)):       # csv_obj[{"age": [3, 5]}]
                print(row_idx)
                row_data = [csv_dict[d] for d in row_idx]
            else:
                raise TypeError()
        else:
            raise IndexError()

        if col_idx is None:
            data = row_data
            title = self.row_data[0]
        elif isinstance(col_idx, str):
            key_idx = self.get_key2idx()[col_idx]  # key
            data = [row[key_idx] for row in row_data]
            title = [col_idx]
        elif isinstance(col_idx, (tuple, list)):
            key2idx = self.get_key2idx()
            keyidxs = [key2idx[key] for key in col_idx]
            data = [[row[key_idx] for key_idx in keyidxs] for row in row_data]
            title = list(col_idx)
        else:
            raise IndexError()
        # ret_data = [title] + data
        # return CSVObject(deepcopy(ret_data))
        return data

    def tolist(self, keepdim=True):
        data = self.row_data[1:]
        if len(data) == 1 and not keepdim:
            data = data[0]
            if len(data) == 1 and not keepdim:
                data = data[0]
        return data

    def __str__(self):
        return "\n".join([",\t".join(row) for row in self.row_data])

    def __repr__(self):
        return self.__str__()

    def shape(self):
        return len(self.row_data)-1, len(self.row_data[0])

    def radd(self, csv_obj, default_value={}):
        other_key2idx = csv_obj.get_key2idx()
        title = self.row_data[0]

        def get_v(data, key):
            if key in other_key2idx:
                return data[other_key2idx[key]]
            elif key in default_value:
                return default_value[key]
            else:
                raise KeyError(key)

        ret_data = [[get_v(data, key) for key in title] for data in csv_obj.row_data[1:]]
        return deepcopy(self.row_data + ret_data)

    def join(self, csv_obj, key):
        csv_dict_b = csv_obj.to_dict(key)
        csv_dict_a = self.to_key_dict(key)
        title_a = self.row_data[0]
        title_b = csv_obj.row_data[0]

        new_csv_data = [title_a + title_b]
        for key in csv_dict_a:
            if key in csv_dict_b:
                new_csv_data.append(csv_dict_a[key] + csv_dict_b[key])
            else:
                new_csv_data.append(csv_dict_a[key] + [None] * len(title_b))
        for key in set(csv_dict_b.keys()) - set(csv_dict_a.keys()):
            new_csv_data.append([None] * len(title_a) + csv_dict_b[key])
        return CSVObject(new_csv_data)


# csv中获得指定行
# 两个csv按照key


if __name__ == '__main__':
    for inst_set in ["a", "b", "c", "d", "f", "m", "I", "zfh"]:
        print(inst_set)
        csv_obj = CSVObject(f"Input/inst_set_input/{inst_set}/{inst_set}_from_json.csv", title_to_lower=False)
        csv_obj_32 = CSVObject(f"Input/inst_set_input_32/{inst_set}/{inst_set}_32.csv", title_to_lower=True)
        rname_upper_32 = csv_obj_32[:, "rname_upper"]
        sub_csv_obj = csv_obj.select_rows_by_key("rname_upper", rname_upper_32)
        sub_csv_obj.save(f"Input/inst_set_input_32/{inst_set}/{inst_set}_from_json_32.csv")

        csv_obj = CSVObject(f"Input/inst_set_input/{inst_set}/{inst_set}_inst_match.csv", title_to_lower=False)
        csv_obj_32 = CSVObject(f"Input/inst_set_input_32/{inst_set}/{inst_set}_32.csv", title_to_lower=True)
        rname_upper_32 = csv_obj_32[:, "rname_upper"]
        sub_csv_obj = csv_obj.select_rows_by_key("rname_upper", rname_upper_32, ignore_miss=True)
        sub_csv_obj.save(f"Input/inst_set_input_32/{inst_set}/{inst_set}_inst_match_32.csv")
