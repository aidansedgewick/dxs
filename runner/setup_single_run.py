import re
import sys
import yaml
from argparse import ArgumentParser
from pathlib import Path

from dxs.runner import ScriptMaker



if __name__ == "__main__":

    #python_script = sys.argv[1]
    #run_dir = sys.argv[2]
    #script_kwargs = sys.argv[3:]
    
    parser = ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("python_script")

    args, s_ak = parser.parse_known_args()

    run_dir = Path(args.run_dir).absolute()
    """
    if run_dir == "default":

        job_list = ["corr", "lf", "uvj", "photoz"]
        jobs = []
        for job in :
            if job in s_ak:
                jobs.append(job)
        if len(jobs) == 0:
            raise ValueError(f"no job found in script arguments from {job_list}")
        job_str = "_".join(x for x in jobs)

        p_obj = re.compile(".*--obj(ect)? ([\s\w]*)")
        obj_str = "_".join(x for x in p_obj.match(s_ak).group(1).split())

        p_tc = re.compile(".*--treecorr-config (.*).yaml")
        tc_str = p_tc.match(s_ak).group(1).split("/")[-1]

        p_opt = re.compile(".*--optical ([\s\w]*)")
        opt_str = "_".join(x for x in p_obj.match(s_ak).group(1).split())        

        run_dir = "_".join([job_str, obj_str, opt_str, tc_str])"""
            
    
    script_maker = ScriptMaker(args.python_script, run_dir)

    header = script_maker.make_header(stdout_name="run")
    header.append("")
    modules = script_maker.make_script_modules()
    modules.append("")

    s_ak_str = " ".join(x for x in s_ak)
    python_cmd = [f"python3 {args.python_script} {s_ak_str}"]

    output_script = run_dir / "run.sh"

    lines = header + modules + python_cmd

    with open(output_script, "w+") as f:
        for line in lines:
            f.writelines(line+"\n")

    print_path = output_script.relative_to(Path.cwd())

    print(f"view script with\n   less {print_path}")

    print(f"\nnow do:\n    \033[35msbatch {print_path}\033[0m")