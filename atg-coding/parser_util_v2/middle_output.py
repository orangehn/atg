from .data_flow_v2 import GraphHandler
import re


class ListReg(object):
    def __init__(self, reg_exp):
        if isinstance(reg_exp, list):
            reg_exp = [re.compile(exp) for exp in reg_exp]
        elif isinstance(reg_exp, str):
            reg_exp = [re.compile(reg_exp)]
        else:
            raise TypeError()
        self.reg_exp = reg_exp

    def match(self, alist):
        pass

    def search(self, alist):
        data = "".join(alist)
        # print(self.reg_exp, re.compile("[(]outs(.*?)[)]"))
        # res = re.compile("[(]outs(.*?)[)]").search(data)
        all_res = []
        for reg_exp in self.reg_exp:
            res = reg_exp.search(data)
            if res is None:
                return
            all_res.append(res)
        return all_res


class GraphIter(object):
    def __init__(self, rname, dataset):
        self.dataset = dataset
        self.all_names = self.collect(rname)

    def collect(self, rname):
        dataset = self.dataset
        r = dataset.records_map[rname]
        rnames = []
        for sr in r.data['supers']:
            if sr[0] in dataset.records_map:
                rnames.extend(self.collect(sr[0]))
        rnames.append(r.data['name'])
        return rnames

    def __getitem__(self, item):
        return self.all_names[item]


class MiddleOut(object):

    @staticmethod
    def do(inst_names, G: GraphHandler):
        # MiddleOut.write_fc_and_fm(*MiddleOut.get_fc_and_fm(inst_names, G))
        dataset = G.dataset

        reg = ListReg(["[(]outs(.*?)[)]", "[(]ins(.*?)[)]"])
        fc = {}
        for inst_name in inst_names:
            names, all_names = [], []
            for i, name in enumerate(GraphIter(inst_name, dataset)):
                all_names.append(name)
                if reg.search(dataset.records_map[name]):
                    names.append(name)
            if len(names) == 0:
                print('[MiddleOut] empty match', inst_name)
                continue
            fc[inst_name] = names

        f = open('outs_ins.tsv', 'w')
        for name in fc:
            f.write(name+"\t")
            assert len(fc[name]) == 1, name
            for n in fc[name]:
                f.write(" ".join(dataset.records_map[n]))
            f.write('\n')
        f.close()
        return fc

    @staticmethod
    def get_fc_and_fm(inst_names, G: GraphHandler):
        dataset = G.dataset

        # count Inst
        InstCount = {}
        for name, r in dataset.records_map.items():
            content = r.data['content']
            count = 0
            for c in content:
                if c[1][0] == 'Inst':
                    count += 1
            InstCount[name] = count

        # print("RVInst16", InstCount['RVInst16'])
        # dataset.records_map['RVInst16'].print()
        fm = {}
        for inst_name in inst_names:
            all_names = []
            max_id, max_v = [], -1
            for i, name in enumerate(GraphIter(inst_name, dataset)):
                all_names.append(name)
                if max_v < InstCount[name]:
                    max_id, max_v = [i], InstCount[name]
                elif max_v == InstCount[name]:
                    max_id.append(i)
            if max_v == 0:
                print("[MiddleOut]: no Inst", inst_name)
                continue
            assert len(max_id) == 1, f"{max_v} {inst_name} {[all_names[i] for i in max_id]}"
            fm[inst_name] = all_names[max_id[0]]

        return MiddleOut.get_fc(inst_names, G), fm

    @staticmethod
    def get_fc(inst_names, G: GraphHandler):
        dataset = G.dataset

        reg = ListReg(["[(]outs(.*?)[)]", "[(]ins(.*?)[)]"])
        fc = {}
        for inst_name in inst_names:
            names, all_names = [], []
            for i, name in enumerate(GraphIter(inst_name, dataset)):
                all_names.append(name)
                if reg.search(dataset.records_map[name]):
                    names.append(name)
            if len(names) == 0:
                print('[MiddleOut] empty match', inst_name)
                continue
            fc[inst_name] = names[-1]
        return fc

    @staticmethod
    def write_fc_and_fm(fc, fm):
        path = 'fm_fc.csv'
        f = open(path, 'w')
        f.write(",".join(['name', 'FC', 'FM']) + '\n')
        for name in set(list(fm.keys()) + list(fc.keys())):
            fm_v = fm.get(name, '#Nan#')
            fc_v = fc.get(name, '#Nan#')
            f.write(','.join([name, fc_v, fm_v]) + '\n')
        f.close()
