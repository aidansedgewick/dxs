import subprocess

from dxs.utilts.misc import check_modules

class StiltsError(Exception):
    pass

class Stilts:
    """
    Class for running stilts tasks.
    kwargs
    
    """    

    def __init__(self, task, flags=None, stilts_exe="stilts", **kwargs):
        check_modules(
        self.stilts_exe = stilts_exe
        self.task = task
        self.flags = flags or {}
        self.stilts_exe = stilts_exe
        self.flags.update(kwargs)
        self.cmd = None

    def run(self):
        if self.cmd is None:
            self.cmd = self.build_cmd()
        status = subprocess.call(self.cmd, shell=True)       
        if status > 0:
            raise StiltsError(f"run: Something went wrong (status={status}).")

    def build_cmd(self, float_precision=6):
        cmd = f"{stilts_exe} {task}"
        for key, value in self.flags.items():
            if isinstance(value, str):
                cmd += f" {key} {value} "
            if isinstance(value, Path):
                cmd += f" {key} {str(value)} "
            if isinstance(value, int):
                cmd += f" {key} {str(value)} "
            if isinstance(value, float):
                cmd += f" {key} {value:.{float_precision}f} "
            else:
                raise StiltsError(f"build_cmd: Don't know how to format type {type(value)}")

    @classmethod
    def tskymatch2_fits(cls, file1, file2, flags=None, stilts_exe="stilts", **kwargs):
        flags = flags or {}
        flags.update(kwargs)
        flags["in1"] = file1
        flags["in2"] = file2
        flags["ifmt1"] = "fits"
        flags["ifmt2"] = "fits"
        return cls("tskymatch2", flags=flags, stilts_exe=stilts_exe)







