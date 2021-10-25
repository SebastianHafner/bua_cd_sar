from pathlib import Path
from fvcore.common.config import CfgNode as _CfgNode

# set the paths
HOME = 'C:/Users/hafne/repos/bua_cd_sar'
DATA = 'C:/Users/hafne/bua_cd_sar/drive'
OUTPUT = 'C:/Users/hafne/bua_cd_sar/output'
SN7_TIMESTAMPS_FILE = ''
SN7_ORBITS_FILE = ''


class CfgNode(_CfgNode):
    """
    The same as `fvcore.common.config.CfgNode`, but different in:

    1. Use unsafe yaml loading by default.
      Note that this may lead to arbitrary code execution: you must not
      load a config file from untrusted sources before manually inspecting
      the content of the file.
    2. Support config versioning.
      When attempting to merge an old config, it will convert the old config automatically.

    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Always allow merging new configs
        self.__dict__[CfgNode.NEW_ALLOWED] = True
        super(CfgNode, self).__init__(init_dict, key_list, True)

    # Note that the default value of allow_unsafe is changed to True
    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = True) -> None:
        loaded_cfg = _CfgNode.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)

        # defaults.py needs to import CfgNode
        self.merge_from_other_cfg(loaded_cfg)


def load_paths() -> CfgNode:
    C = CfgNode()
    C.HOME = HOME
    C.DATA = DATA
    C.OUTPUT = OUTPUT
    C.SN7_TIMESTAMPS_FILE = SN7_TIMESTAMPS_FILE
    C.SN7_ORBITS_FILE = SN7_ORBITS_FILE
    return C.clone()


def setup_directories():
    dirs = load_paths()

    # inference dir
    output_dir = Path(dirs.OUTPUT)
    output_dir.mkdir(exist_ok=True)

    # change variables
    cv_dir = Path(dirs.OUTPUT) / 'change_variables'
    cv_dir.mkdir(exist_ok=True)

    # change maps
    cm_dir = Path(dirs.OUTPUT) / 'change_maps'
    cm_dir.mkdir(exist_ok=True)

    # plots
    plots_dir = Path(dirs.OUTPUT) / 'plots'
    plots_dir.mkdir(exist_ok=True)


if __name__ == '__main__':
    setup_directories()
