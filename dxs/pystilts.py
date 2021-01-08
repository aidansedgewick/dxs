import logging
import subprocess

from dxs.utils.misc import check_modules, format_flags

logger = logging.getLogger("stilts_wrapper")

class StiltsError(Exception):
    pass

table_processing_commands = (
    "tcopy tpipe tmatch2 tskymatch2 tmatch1 tmatchn tjoin tcube".split()
    + "tcat tcatn tmulti tmultin tloop".split()
)
plotting_commands = "plot2plane plot2sky plot2cube plot2sphere plot2time".split()
vot_commands = "votcopy votlint".split()
vo_commands = (
    "cone tapquery tapresume tapskymatch cdsskymatch taplint".split()
    + "datalinklint regquery coneskymatch".split()
)
skypix_commands = "tskymap pixfoot pixsample".split()
sql_commands = "sqlskymatch sqlclient sqlupdate".split()
misc_commands = "server calc funcs".split()
all_commands = (
    table_processing_commands + plotting_commands + vot_commands + vo_commands
    + skypix_commands + sql_commands + misc_commands
)


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

    """
    docs_url = "http://www.star.bris.ac.uk/~mbt/stilts/"

    def __init__(self, task, flags=None, stilts_exe="stilts", **kwargs):
        check_modules("stilts")
        self._task_check(task)
        self.stilts_exe = stilts_exe
        self.task = task
        self.flags = flags or {}
        self.stilts_exe = stilts_exe
        self.flags.update(kwargs)
        self.cmd = None
        
        if "out" not in flags:
            flags

    @staticmethod
    def _task_check(task):
        if task not in all_commands:
            raise StiltsError("task {self.task} not recognised")
        if task not in table_processing_commands:
            logger.warn(f"task {self.task} behaviour not tested with this wrapper...")

    def run(self):
        if self.cmd is None:
            self.cmd = self.build_cmd()
        status = subprocess.call(self.cmd, shell=True)       
        if status > 0:
            error_msg = (
                f"run: Something went wrong (status={status}).\n "
                f"check docs? {docs_url}/http://www.star.bris.ac.uk/~mbt/stilts/sun256/{self.task}.html"
            )
            raise StiltsError(f"run: Something went wrong (status={status}).")

    def build_cmd(self, float_precision=6):
        cmd = f"{stilts_exe} {task} "
        flags = format_flags(self.flags, capitalise=False, float_precision=float_precision)
        cmd += " ".join(f"{k}={v}" for k, v in flags.items())
        self.cmd = cmd

    @classmethod
    def tskymatch2_fits(
        cls, file1, file2, output, ra=None, dec=None, flags=None, 
        stilts_exe="stilts", **kwargs
    ):
        flags = flags or {}
        flags["in1"] = file1
        flags["in2"] = file2
        flags["ifmt1"] = "fits"
        flags["ifmt2"] = "fits"
        flags["omode"] = "out"
        flags["ofmt"] = "fits"
        flags["out"] = output
        if ra is not None:
            flags["ra1"] = ra
            flags["ra2"] = ra
        if dec is not None:
            flags["dec1"] = dec
            flags["dec2"] = dec
        return cls("tskymatch2", flags=flags, stilts_exe=stilts_exe)

    @classmethod
    def tskymatch2_(self):
        pass







