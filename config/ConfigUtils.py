import copy
def print_config(cfg):
    # iterator = itertools.groupby(cfg.to_dict(), lambda keyvalue: keyvalue[0].split(".")[0])
    cfg_dict = cfg.to_dict()
    str = ''
    right_tabs = max([len(name__) for name__, key__ in copy.deepcopy(cfg_dict).items() if type(key__) is not dict]) + 1
    for name, key in cfg_dict.items():
        if type(key) is not dict:
            str += f'{name:{right_tabs}}: {type(key).__name__}\t = {key} \n'
    for name, key in cfg_dict.items():
        if type(key) is dict:
            str += f"{name.upper()} \n"
            right_tabs = max(
                [len(name__) for name__, key__ in copy.deepcopy(cfg_dict).items() if type(key__) is not dict]) + 1
            for name_, key_ in key.items():
                str += f"\t {name_:{right_tabs}}: {type(key_).__name__}\t = {key_} \n"
    # print(str)
    # sys.exit()

    print(str)