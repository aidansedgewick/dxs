import logging
import subprocess
import yaml
from pathlib import Path

from dxs.utils.misc import check_modules, format_flags, create_file_backups

from dxs import paths

logger = logging.getLogger("stilts_wrapper")

class StiltsError(Exception):
    pass

def _load_known_tasks():
    known_tasks_path = Path(__file__).absolute().parent / "known_tasks.yaml"
    with open(known_tasks_path, "r") as f:
        known_tasks= yaml.load(f, Loader=yaml.FullLoader)
        known_tasks["all_tasks"] = [
            task for task_type in known_tasks.values() for task in task_type
        ]
    return known_tasks

docs_url = "http://www.star.bris.ac.uk/~mbt/stilts/"

class Stilts:
    """
    Class for running stilts tasks, http://www.star.bris.ac.uk/~mbt/stilts/
    Can provide cmd line flags as dictionary in flags, or as kwargs.
    kwarg value overwrites flags value.
    eg.
    >>> stilts = Stilts("tskymatch2", flags={"error": 0.2}, error=0.3)
    >>> stilts.run()
    will run tskymatch2 with error=0.3

    for the base class, NO command line flags are assumed by default.

    Parameters
    ----------
    task
        stilts task - see known stilts tasks. 
    flags
        command line flags.
    stilts_exe
        if your `stilts` executable is not in the path (ie, can't run "stilts" 
        from the command line), provide path to executable here.
    kwargs
        extra kwargs to pass to stilts

    """


    def __init__(self, task, flags=None, stilts_exe="stilts", **kwargs):
        check_modules("stilts")
        self._task_check(task)
        self.stilts_exe = stilts_exe
        self.task = task
        self.flags = flags or {}
        self.stilts_exe = stilts_exe
        self.flags.update(kwargs)
        print("FLAGS ARE", self.flags)
        self.cmd = None
        
        if "out" not in flags:
            flags["out"] = Path.cwd() / f"{task}.out"

    @staticmethod
    def _task_check(task):
        known_tasks = _load_known_tasks()
        if task not in known_tasks["all_tasks"]:
            print(f"known_tasks are", known_tasks["all_tasks"])
            raise StiltsError(f"task {task} not recognised")
        if task not in known_tasks["table_processing_commands"]:
            logger.warn(f"task {task} behaviour not tested with this wrapper...")

    def run(self, strict=True):
        if self.cmd is None:
            self.build_cmd()
        print("\n")
        logger.info(f"RUN CMD:\n  {self.cmd}")
        status = subprocess.call(self.cmd, shell=True)
        if strict:    
            if status > 0:
                print()
                error_msg = (
                    f"run: Something went wrong (status={status}).\n"
                    + f"check docs? {docs_url}sun256/{self.task}.html"
                )
                raise StiltsError(error_msg)
        return status

    def build_cmd(self, float_precision=6):
        cmd = f"{self.stilts_exe} {self.task} "
        flags = format_flags(self.flags, capitalise=False, float_precision=float_precision)
        cmd += " ".join(f"{k}={v}" for k, v in flags.items())
        self.cmd = cmd

    @classmethod
    def tskymatch2_fits(
        cls, file1_path, file2_path, output_path, ra=None, dec=None, flags=None, 
        stilts_exe="stilts", **kwargs
    ):
        if file1_path == file2_path:
            raise StiltsError(f"tskymatch2: file1 == file2!?! {file1_path} {file2_path}")
        if file1_path == output_path and file1_path.exists():
            new_paths = create_file_backups(file1_path, paths.temp_data_path)
            file1_path = new_paths[0] # filebackups returns list.
        elif file2_path == output_path and file2_path.exists():
            new_paths = create_file_backups(file2_path, paths.temp_data_path)
            file2_path = new_paths[0]

        flags = flags or {}
        flags["in1"] = file1_path
        flags["in2"] = file2_path
        flags["ifmt1"] = "fits"
        flags["ifmt2"] = "fits"
        flags["omode"] = "out"
        flags["ofmt"] = "fits"
        flags["out"] = output_path
        if ra is not None:
            flags["ra1"] = ra
            flags["ra2"] = ra
        if dec is not None:
            flags["dec1"] = dec
            flags["dec2"] = dec
        return cls("tskymatch2", flags=flags, stilts_exe=stilts_exe, **kwargs)

    @classmethod
    def tmatch2_fits(
        cls, file1, file2, output_path, flags=None, **kwargs
    ):
        raise NotImplementedError
        if file1 == file2:
            raise StiltsError(f"tskymatch2: file1 == file2!?! {file1} {file2}")
        if file1 == output and file1.exists():
            new_paths = create_file_backups(file1, paths.temp_data_path)
            file1 = new_paths[0] # filebackups returns list.
        elif file2 == output and file2.exists():
            new_paths = create_file_backups(file2, paths.temp_data_path)
            file2 = new_paths[0]

        flags = flags or {}
        flags["in1"] = file1
        flags["in2"] = file2
        flags["ifmt1"] = "fits"
        flags["ifmt2"] = "fits"
        flags["omode"] = "out"
        flags["ofmt"] = "fits"
        flags["out"] = output
        
        return cls("tmatch2", flags=flags, stilts_exe=stilts_exe, **kwargs)        

    @classmethod
    def tcat_fits(
        cls, table_list, output_path, flags=None, stilts_exe="stilts", **kwargs
    ):
        flags = flags or {}
        flags["in"] = "\"" + " ".join(str(t) for t in table_list) + "\""
        flags["ifmt"] = "fits"
        flags["omode"] = "out"
        flags["ofmt"] = "fits"
        flags["out"] = output_path

        return cls("tcat", flags=flags, stilts_exe=stilts_exe, **kwargs)
    
    @classmethod
    def tmatch1_sky_fits(
        cls, table_path, output_path, ra, dec, error,
        flags=None, stilts_exe="stilts", **kwargs
    ):
        if table_path == output_path:
            new_paths = create_file_backups(table_path, paths.temp_data_path)
            table_path = new_paths[0]

        flags = flags or {}
        flags["in"] = table_path
        flags["ifmt"] = "fits"
        flags["omode"] = "out"
        flags["ofmt"] = "fits"
        flags["out"] = output_path
        flags["matcher"] = "sky"
        flags["values"] = "\"" + f"{ra} {dec}" + "\""
        flags["params"] = f"{error:.8f}"
        flags["action"] = "identify"

        return cls("tmatch1", flags=flags, stilts_exe=stilts_exe, **kwargs)        








