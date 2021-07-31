import re
import yaml
from itertools import count
from pathlib import Path

from dxs import paths

default_system_config_path = paths.config_path / "system_config.yaml"
run_config_default_path = paths.config_path / "run_config.yaml"

class ScriptMaker:
    _script_number = count()

    def __init__(
        self, python_script_path, base_dir,
        config=None,
        system_config_path=default_system_config_path,
        job_name=None
    ):
        self.config = config or {}
        self.system_config = self._load_configuration(system_config_path)
        self.python_script_path = Path(python_script_path)
        self.base_dir = base_dir
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.stdout_dir = self.base_dir / "stdout"
        self.stdout_dir.mkdir(exist_ok=True, parents=True)
        self.job_name = job_name or "mosaic"
        self.cpus_per_job = min(
            self.system_config.get("cpus_per_node", 999),
            self.config.get("cpus_per_job", 999)
        )
        if self.cpus_per_job == 999:
            raise ValueError(
                f"either set \"cpus_per_node\" in {system_config_path}, "
                + f"or \"cpus_per_job\" in config"
            )

    def _load_configuration(self, system_config_path):
        with open(system_config_path, "r") as f:
            system_config = yaml.load(f, Loader=yaml.FullLoader)
        return system_config

    def write_all_scripts(self, args_iterable, kwargs_iterable):
        script_paths = []
        for ii, (args, pykwargs) in enumerate(zip(args_iterable, kwargs_iterable)):
            script_path = self.write_script(python_args=args, python_kwargs=pykwargs)
            script_paths.append(script_path)
            if ii == 0:
                try:
                    print_path = script_path.relative_to(Path.cwd())
                except:
                    print_path = script_path
                print(f"running scripts written to eg.\n    {print_path}")
                try:
                    script_text = open(script_path, "r").read().replace("\n", "")
                    stdout_res = re.search("#SBATCH -o(.+?)#", script_text).groups(0)[0].strip()
                    try:
                        print_stdout_path = Path(stdout_res).relative_to(Path.cwd())
                    except:
                        print_stdout_path = Path
                except:
                    print_stdout_path = None

        all_scripts_path = self.base_dir / "submit_all.sh"
        submit_all_script = self.make_submit_all_script(script_paths)
        #if print_stdout_path is not None:
        submit_all_script += [f"printf 'monitor with eg.\\n    less {print_stdout_path}\\n'"]
        with open(all_scripts_path, "w") as f:
            for line in submit_all_script:
                f.write(line + "\n")
        try:
            print_path = all_scripts_path.relative_to(Path.cwd())
        except:
            print_path = all_scripts_path
        print(f"submit all scripts with:\n    \033[035mbash {print_path}\033[0m")

    def write_script(
        self, python_args=None, python_kwargs=None, script_number=None,
    ):
        if script_number is None:
            script_number = next(ScriptMaker._script_number)
        script_dir = self.base_dir / f"run_{script_number:03d}"
        script_dir.mkdir(exist_ok=True, parents=True)
        script_path = script_dir / f"run_{script_number:03d}.sh"
        python_args = python_args or []
        python_kwargs = python_kwargs or {}
        for k, v in python_kwargs.items():
            if v is None:
                python_kwargs[k] = ""
        header = self.make_header(
            stdout_name=script_path.stem, 
            job_name=self.job_name, 
            script_number=script_number
        )
        modules = self.make_script_modules()
        python_command = self.make_python_command(python_args, python_kwargs)
        submission_script = header + modules + [python_command]

        with open(script_path, "w") as f:
            for line in submission_script:
                f.write(line + "\n")
        return script_path        
        
    def make_header(self, stdout_name=None, job_name=None, script_number=None):
        if script_number is None:
            script_number = 0
        if job_name is None:
            job_name = f"mosaic"
        max_time = self.system_config["max_time"]
        scheduler = self.system_config["scheduler"]
        queue = self.system_config["queue"]
        account = self.system_config.get("account", None)
        extra_header_lines = self.system_config.get("extra_header_lines", None)
        if stdout_name is None:
            stdout_name = f"run_{script_number:03d}"
        stdout_path = self.stdout_dir / stdout_name
        if scheduler == "slurm":
            header = [
                "#!/bin/bash -l",
                "",
                f"#SBATCH --ntasks {self.cpus_per_job}",
                f"#SBATCH -J {job_name[0:4]}_{script_number:03d}",
                f"#SBATCH -p {queue}",
                f"#SBATCH -o {stdout_path}.out",
                f"#SBATCH -e {stdout_path}.err",
                f"#SBATCH -t {max_time}",
            ]
            if account:
                header.append(f"#SBATCH -A {account}")
        elif scheduler == "pbs":
            header = [
                "#!/bin/bash -l",
                "",
                f"#PBS -N {self.job_name[0:4]}_{script_number:03d}",
                f"#PBS -l procs={self.cpus_per_job}",
                f"#PBS -l walltime={max_time}",
                f"#PBS -q {queue}",
                f"#PBS -A {account}",
                f"#PBS -o {stdout_path}.out",
                f"#PBS -e {stdout_path}.err",
            ]
        elif scheduler == "lsf":
            header = [
                "#!/bin/bash -l",
                "",
                '#BSUB -R "span[ptile={}]"'.format(
                    min(
                        self.cpus_per_job,
                        self.system_config["cores_per_node"]
                    )
                ),
                f"#BSUB -n {self.cpus_per_job}",
                f"#BSUB -J {self.job_name[0:4]}_{script_number:03d}",
                f"#BSUB -q {queue}",
                f"#BSUB -P {account}",
                f"#BSUB -o {stdout_path}.out",
                f"#BSUB -e {stdout_path}.err",
                f"#BSUB -x",
                f"#BSUB -W {max_time}",
            ]
        else:
            raise ValueError(f"Scheduler {scheduler} not yet supported.")
        if extra_header_lines:
            header += extra_header_lines
        return header

    def make_script_modules(self,):
        modules = []
        system_modules = self.system_config.get("modules", None)
        if system_modules is None:
            print("Your system config has no MODULES to load/source ?!?!")
            return modules
        modules = [ purge_cmd for purge_cmd in system_modules.get("purge", []) ]
        modules = modules + [
           f"module load {mod}" for mod in system_modules.get("modules_to_load", [])
        ]
        modules = modules + [
            f"source {mod}" for mod in system_modules.get("modules_to_source", [])
        ]
        modules = modules + [
            f"{mod}" for mod in system_modules.get("extra", [])
        ]
        return modules

    def make_python_command(self, python_args, python_kwargs):
        arg_str = " ".join(f"{arg}" for arg in python_args)
        kwarg_str = " ".join(f"--{k} {v}" for k, v in python_kwargs.items())
        python_command = f"python3 -u {self.python_script_path} {arg_str} {kwarg_str}"
        return python_command

    def make_submit_all_script(self, script_paths):
        script = ["#!/bin/bash -l \n"]
        scheduler = self.system_config["scheduler"]
        if scheduler == "slurm":
            submission_command = "sbatch"
        elif scheduler == "pbs":
            submission_command = "qsub"
        elif scheduler == "lsf":
            if self.system_config["name"] == "hartree":
                # need arrow to direct script
                submission_command = "bsub <"
            else:
                submission_command = "bsub"

        else:
            raise ValueError(f"Scheduler {scheduler} not yet supported.")
        for path in script_paths:
            script += [f"{submission_command} {path}"]
        return script



    
