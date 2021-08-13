import datetime
import yaml
from argparse import ArgumentParser
from itertools import product
from pathlib import Path


from dxs.utils.misc import tile_parser

from runner import ScriptMaker

from dxs import paths


system_config_path = paths.config_path / "system_config.yaml"
with open(system_config_path, "r") as f:
    system_config = yaml.load(f, Loader=yaml.FullLoader)


def fix_placeholders(config, placeholders=None):
    if placeholders is None:
        placeholders = {}
    for k, v in config.items():
        if type(v) is str:
            if "@" in v:
                spl = v.split("/")
                rpl = [placeholders[x[1:]] if "@" in x else x for x in spl]
                join = Path(*tuple(rpl))
                config[k] = str(join)
            placeholders[k] = config[k]
        elif isinstance(v,dict):
            config[k] = fix_placeholders(v, placeholders=placeholders)
    return config

def read_run_config(run_config_path):
    with open(run_config_path, "r") as f:
        run_config = yaml.load(f, Loader=yaml.FullLoader)
    run_config = fix_placeholders(run_config)
    for k, v in run_config.get("kwargs", {}).items():
        if v == "None":
            run_config["kwargs"][k] = None
    return run_config
    
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", action="store", required=True, 
        help="config required. see dxs/runner/blank_config.yaml for an example")

    args = parser.parse_args()
    
    run_config_path = Path(args.config)
    run_config = read_run_config(run_config_path)
    run_name = run_config.get("run_name", None)
    datestr = datetime.datetime.today().strftime("%y%m%d")
    if run_name is None:
        run_name = f"{run_config_path.stem}_{datestr}"
        print(f"set run_name to {run_name}")
    run_name = run_name.replace("$date", datestr)

    base_dir = paths.runner_path / f"runs/{run_name}"
    base_dir.mkdir(exist_ok=True, parents=True)
    
    python_script_path = Path(run_config["python_script_path"]).absolute()
    if not python_script_path.exists():
        print(f"\033[31;1mWARN:\033[0m {python_script_path} not found.")
    job_name = run_config.get("job_name", None)
    script_maker = ScriptMaker(python_script_path, base_dir=base_dir, job_name=job_name)

    arg_list = []
    for k, v in run_config.get("args",{}).items():
        if type(v) is str:
            if not v[0] == "@":
                init_list = v.split(",")
                proc_list = []
                for x in init_list:
                    if "-" in x:
                        spl = x.split("-")
                        proc_list.extend([i for i in range(int(spl[0]), int(spl[1])+1)])
                    else:
                        proc_list.append(int(x) if x.isnumeric() else x)     
                arg_list.append(proc_list)
            if v[0] == "@":
                v = v[1:]
                arg_list.append([v])
        else:
            arg_list.append([v])

    arg_tuple = tuple(arg_list)
    combinations = product(*arg_tuple)
    combinations = [x for x in combinations] # no genexpr!

    kwargs = run_config.get("kwargs", {})

    per_run_kwargs = run_config.get("per_run_kwargs", {})
    if len(per_run_kwargs) > 0:
        kwargs_list = [kw for kw in per_run_kwargs]
        for kw in kwargs_list:
            
            kw.update(kwargs)
        print(kwargs_list)
        if len(combinations) == 1:
            combinations = [() for _ in kwargs_list]
        elif len(kwargs_list) != len(combinations):
            raise ValueError("diff num. combinations to per_run_kwargs!")   
    else:
        kwargs_list = [kwargs for _ in combinations]


    print(f"There are {len(combinations)} combinations")

    script_maker.write_all_scripts(combinations, kwargs_list)








