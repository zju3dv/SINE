import addict


def update_config(config, unknown):
    # update config given args
    for idx, arg in enumerate(unknown):
        if arg.startswith("--"):
            if (":") in arg:
                k1, k2 = arg.replace("--", "").split(":")
                argtype = type(config[k1][k2])
                if argtype == bool:
                    v = unknown[idx + 1].lower() == "true"
                else:
                    if config[k1][k2] is not None:
                        v = type(config[k1][k2])(unknown[idx + 1])
                    else:
                        v = unknown[idx + 1]
                print(f"[Info] Changing unknown {k1}:{k2} ---- {config[k1][k2]} to {v}")
                config[k1][k2] = v
            else:
                k = arg.replace("--", "")
                v = unknown[idx + 1]
                argtype = type(config[k])
                print(f"[Info] Changing unknown {k} ---- {config[k]} to {v}")
                config[k] = v

    return config


class ForceKeyErrorDict(addict.Dict):
    def __missing__(self, name):
        raise KeyError(name)
