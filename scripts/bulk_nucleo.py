import pandas as pd
import numpy as np
import argparse
import os
from osnap import *

if __name__ == "__main__":

    # Parses command line arguments so that we can limit which models we're submitting jobs for
    parser = argparse.ArgumentParser(description='Run nucleosynthesis calculations')
    parser.add_argument('-t', '--tracer-count', nargs='+', type=int, default=[100],
                        help='Number of tracers (can specify multiple values for multiple runs), default: 100')
    parser.add_argument('-m', '--masses', nargs='+', type=str, required=True,
                        help='ZAMS masses (can specify multiple values, or use "all" for all masses)')
    parser.add_argument('-a', '--alphas', nargs='*', type=str, default=["1.25"],
                        help='Alpha values (can specify multiple values, default: "1.25")')
    parser.add_argument('-f', '--force-reload', action='store_true',
                        help='Forces all tracer data, nucleosynthesis, and stiching to be rerun even if there is existing usable data cached.')
    args = parser.parse_args()

    # Read in the model library created by the model_library.py script
    # This file contains all Sukhbold 2016 models that have been run through STIR and did explode
    models = pd.read_csv(f"{config.main_data_directory}/model_library.csv", dtype={"mass": str, "alpha": str, "folder_path": str, "model_name": str})

    # If "all" is specified, use all masses that are available from sukhbold 2016
    if args.masses[0] == "all":
        masses = np.array(config.all_masses, dtype=str)
    else:
        masses = np.array(args.masses, dtype=str)

    # Start a job for all requested models, excluding ones that we already know did not explode
    for index, row in models.iterrows():
        if str(row["mass"]) in masses and str(row["alpha"]) in args.alphas:
            for tracer_count in args.tracer_count:

                # Setting environment variables for the job so that we can use one submission script for all jobs
                os.environ["MASS"] = str(row["mass"])
                os.environ["ALPHA"] = str(row["alpha"])
                os.environ["NUM_TRACERS"] = str(tracer_count)

                # Setting the job name and output files here since you can't use arguments in SBATCH lines
                job_name = f"nucleo_run_n{tracer_count}_a{row['alpha']}_m{row['mass']}"
                output_file = f"nucleo_runs/nucleo_run_n{tracer_count}_a{row['alpha']}_m{row['mass']}_%j.out"

                # Submits the job
                print(f"Submitting job for mass {row['mass']}, alpha {row['alpha']}, and {tracer_count} tracers")
                os.system(f'sbatch --job-name={job_name} --output={output_file} run_quick_nucleo.slurm')