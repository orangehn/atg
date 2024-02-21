from enum import Enum
import traceback


class Flag:
    parser_record = False


class LOG(Enum):
    ERROR = 0
    WARN = 1
    INFO = 2
    STEP_INFO = 3
    DEBUG = 4

    log_level = ERROR  # INFO #只会输入包含它及以上

    @staticmethod
    def log(out_level, *args, **kwargs):
        if LOG.log_level.value >= out_level.value:
            print("[{}]:".format(out_level), *args, **kwargs)

    @staticmethod
    def debug(*args, **kwargs):
        fs = traceback.extract_stack(limit=2)[0]
        args = list(args) + [f'({fs})']
        LOG.log(LOG.DEBUG, *args, **kwargs)

    @staticmethod
    def info(*args, **kwargs):
        fs = traceback.extract_stack(limit=2)[0]
        args = list(args) + [f'({fs})']
        LOG.log(LOG.INFO, *args, **kwargs)

    @staticmethod
    def warn(*args, **kwargs):
        fs = traceback.extract_stack(limit=2)[0]
        args = list(args) + [f'({fs})']
        LOG.log(LOG.WARN, *args, **kwargs)

    @staticmethod
    def fprint(flag, *args, **kwargs):
        if flag:
            print("[FPRINT]:", *args, **kwargs)


# class List2d(object):
#     def __init__.py(self, list2d):
#         self.data = list2d
#         self.length = [len(d) for d in list2d]
#
#     def get_2d_index(self, idx):
#         i, j = 0, idx
#         while j > self.length[i]:
#             j -= self.length[i]
#             i += 1
#         return i, j
#
#     def get_1d_index(self, i, j):
#         idx = j
#         for ii in range(i):
#             idx += self.length[ii]
#         return idx
#
#     def next_idx(self, i, j):
#         if i >= len(self.data):
#             raise ValueError("i should <", len(self.data))
#         if j < len(self.data[i]) - 1:
#             return i, j + 1
#         elif j == len(self.data[i]) - 1:
#             return i + 1, 0
#         else:
#             raise ValueError("j should <", len(self.data[i]))
