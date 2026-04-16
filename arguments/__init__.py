from argparse import ArgumentParser, Namespace
import os
import sys

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none: bool = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]

            t = type(value)
            value = value if not fill_none else None

            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, "-" + key[0], default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, "-" + key[0], default=value, type=t
                    )

            else:
                if t == bool:
                    group.add_argument(
                        "--" + key, default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, default=value, type=t
                    )

    def extract(self, args):
        group = GroupParams()
        for k, v in vars(args).items():
            if k in vars(self) or ("_" + k) in vars(self):
                setattr(group, k, v)
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser: ArgumentParser, sentinel: bool = False):
        self._source_path = "./dataset/measured_LuViRA_100by100"
        self._model_path = ""
        self.data_device = "cuda"
        self.eval = False

        # beam-by-subcarrier only
        self.num_beams = 100
        self.num_subcarriers = 100

        # local 2D splat
        self.plane_support_radius = 1
        self.plane_init_sigma_beam = 0.70
        self.plane_init_sigma_subcarrier = 0.70
        self.plane_min_sigma = 0.25
        self.plane_max_sigma = 1.20

        self.use_dynamic_center = True
        self.use_dynamic_sigma = True
        self.center_shift_max_beam = 1.5
        self.center_shift_max_subcarrier = 1.5
        self.sigma_log_shift_max_beam = 0.5
        self.sigma_log_shift_max_subcarrier = 0.5

        self.init_mode = "random"
        self.vertices_path = ""

        super().__init__(parser, "Model Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        if getattr(g, "source_path", ""):
            g.source_path = os.path.abspath(g.source_path)
        if getattr(g, "model_path", ""):
            g.model_path = os.path.abspath(g.model_path)
        if getattr(g, "vertices_path", ""):
            g.vertices_path = os.path.abspath(g.vertices_path)
        return g

class OptimizationParams(ParamGroup):
    def __init__(self, parser: ArgumentParser):
        self.iterations = 200_000
        self.optimizer_type = "default"

        # plane rasterizer params
        self.plane_center_lr = 0.01
        self.plane_sigma_lr = 0.003
        self.opacity_lr = 0.025
        self.opacity_lr_final = 0.003

        # dynamic gain head only
        self.dynamic_gain_lr = 0.001
        self.dynamic_gain_lr_final = 0.0001
        self.dynamic_center_lr = 0.001
        self.dynamic_center_lr_final = 0.0001
        self.dynamic_sigma_lr = 0.001
        self.dynamic_sigma_lr_final = 0.0001

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser: ArgumentParser):

    args_cmdline = parser.parse_args(sys.argv[1:])
    cfgfile_string = "Namespace()"

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath, "r", encoding="utf-8") as cfg_file:
            print("Config file found:", cfgfilepath)
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError, AttributeError):
        print("Config file not found.")
        pass

    args_cfgfile = eval(cfgfile_string, {"Namespace": Namespace}, {})
    merged_dict = vars(args_cfgfile).copy()

    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v

    return Namespace(**merged_dict)