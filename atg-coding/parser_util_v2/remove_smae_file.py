def get_file_str(filename):
    data = open(filename).read()
    data = [d for d in data if not d.isspace()]
    data = " ".join(data)
    return data


def get_files(dir):
    import os
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
    return filenames

from collections import defaultdict
filenames = get_files(dir='../Input/MIPS')
content_map = defaultdict(list)
for filename in filenames:
    filestr = get_file_str(filename)
    content_map[filestr].append(filename)
for k, v in content_map.items():
    print(v)
