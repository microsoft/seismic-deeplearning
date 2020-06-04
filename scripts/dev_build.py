#!/usr/bin/env python3
""" Please see the def main() function for code description."""
import time

""" libraries """

import numpy as np
import os
import sys
import yaml
import subprocess

np.set_printoptions(linewidth=200)
import logging

# toggle to WARNING when running in production, or use CLI
logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger().setLevel(logging.WARNING)
import argparse

parser = argparse.ArgumentParser()

""" useful information when running from a GIT folder."""
myname = os.path.realpath(__file__)
mypath = os.path.dirname(myname)
myname = os.path.basename(myname)

def main(args):
    """

    Runs main build jobs on your local VM. By default setup job is not run,
    add --setup to run it (destroys existing environment and creates a new one, along with all the data)

    """

    logging.info("loading data")

    with open(args.file) as file:

        list = yaml.load(file, Loader=yaml.FullLoader)
        logging.info(f"Loaded {file}")

        # run single job
        job_names = [x["job"] for x in list["jobs"]] if not args.job else args.job.split(',')

        if not args.setup and "setup" in job_names:
            job_names.remove("setup")

        job_list = list["jobs"]

        # copy existing environment
        current_env = os.environ.copy()
        # modify for conda to work
        # TODO: not sure why on DS VM this does not get picked up from the standard environment
        current_env["PATH"] = PATH_PREFIX+":"+current_env["PATH"]

        for job in job_list:
            job_name = job["job"]
            if job_name not in job_names:
                continue

            bash = job["steps"][0]["bash"]

            logging.info(f"Running job {job_name}")

            try:
                tic = time.perf_counter()
                completed = subprocess.run(
                    # 'set -e && source activate seismic-interpretation && which python && pytest --durations=0 cv_lib/tests/',
                    bash,
                    check=True,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    executable=current_env["SHELL"],
                    env=current_env,
                    cwd=os.getcwd()
                )
                toc = time.perf_counter()
                print(f"Job time took {(toc-tic)/60:0.2f} minutes")
            except subprocess.CalledProcessError as err:
                logging.info(f'ERROR: \n{err}')
                decoded_stdout = err.stdout.decode('utf-8')
                log_file = "dev_build.latest_error.log"
                logging.info(f"Have {len(err.stdout)} output bytes in {log_file}")
                with open(log_file, 'w') as log_file:
                    log_file.write(decoded_stdout)
                sys.exit()
            else:
                logging.info(f"returncode: {completed.returncode}")
                logging.info(f"Have {len(completed.stdout)} output bytes: \n{completed.stdout.decode('utf-8')}")

    logging.info(f"Everything ran! You can try running the same jobs {job_names} on the build VM now")

""" GLOBAL VARIABLES """
PATH_PREFIX = "/data/anaconda/envs/seismic-interpretation/bin:/data/anaconda/bin"

parser.add_argument(
    "--file", help="Which yaml file you'd like to read which specifies build info", type=str, required=True
)
parser.add_argument(
    "--job", help="CVS list of the job names which you would like to run", type=str, default=None, required=False
)
parser.add_argument("--setup", help="Add setup job", action="store_true")

""" main wrapper with profiler """
if __name__ == "__main__":
    main(parser.parse_args())

# pretty printing of the stack
"""
  try:
    logging.info('before main')
    main(parser.parse_args())
    logging.info('after main')
  except:
    for frame in traceback.extract_tb(sys.exc_info()[2]):
      fname,lineno,fn,text = frame
      print ("Error in %s on line %d" % (fname, lineno))
"""
# optionally enable profiling information
#  import cProfile
#  name = <insert_name_here>
#  cProfile.run('main.run()', name + '.prof')
#  import pstats
#  p = pstats.Stats(name + '.prof')
#  p.sort_stats('cumulative').print_stats(10)
#  p.sort_stats('time').print_stats()
