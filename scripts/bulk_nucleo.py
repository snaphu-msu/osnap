import pandas as pd
import numpy as np
import argparse
import os

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
    models = pd.read_csv("../data/model_library.csv", dtype={"mass": str, "alpha": str, "folder_path": str, "model_name": str})

    # If "all" is specified, use all masses that are available from sukhbold 2016
    if args.masses[0] == "all":
        masses = np.array(["9.0", "9.25", "9.5", "9.75", "10.0", "10.25", "10.5", "10.75", "11.0", "11.25", "11.5", "11.75", 
                    "12.0", "12.25", "12.5", "12.75", "13.0", "13.1", "13.2", "13.3", "13.4", "13.5", "13.6", "13.7", 
                    "13.8", "13.9", "14.0", "14.1", "14.2", "14.3", "14.4", "14.5", "14.6", "14.7", "14.8", "14.9", 
                    "15.0", "15.1", "15.2", "15.3", "15.4", "15.5", "15.6", "15.7", "15.8", "15.9", "16.0", "16.1", 
                    "16.2", "16.3", "16.4", "16.5", "16.6", "16.7", "16.8", "16.9", "17.0", "17.1", "17.2", "17.3", 
                    "17.4", "17.5", "17.6", "17.7", "17.8", "17.9", "18.0", "18.1", "18.2", "18.3", "18.4", "18.5", 
                    "18.6", "18.7", "18.8", "18.9", "19.0", "19.1", "19.2", "19.3", "19.4", "19.5", "19.6", "19.7", 
                    "19.8", "19.9", "20.0", "20.1", "20.2", "20.3", "20.4", "20.5", "20.6", "20.7", "20.8", "20.9", 
                    "21.0", "21.1", "21.2", "21.3", "21.4", "21.5", "21.6", "21.7", "21.8", "21.9", "22.0", "22.1", 
                    "22.2", "22.3", "22.4", "22.5", "22.6", "22.7", "22.8", "22.9", "23.0", "23.1", "23.2", "23.3", 
                    "23.4", "23.5", "23.6", "23.7", "23.8", "23.9", "24.0", "24.1", "24.2", "24.3", "24.4", "24.5", 
                    "24.6", "24.7", "24.8", "24.9", "25.0", "25.1", "25.2", "25.3", "25.4", "25.5", "25.6", "25.7", 
                    "25.8", "25.9", "26.0", "26.1", "26.2", "26.3", "26.4", "26.5", "26.6", "26.7", "26.8", "26.9", 
                    "27.0", "27.1", "27.2", "27.3", "27.4", "27.5", "27.6", "27.7", "27.8", "27.9", "28.0", "28.1", 
                    "28.2", "28.3", "28.4", "28.5", "28.6", "28.7", "28.8", "28.9", "29.0", "29.1", "29.2", "29.3", 
                    "29.4", "29.5", "29.6", "29.7", "29.8", "29.9", "30.0", "31", "32", "33", "35", "40", "45", "50", 
                    "55", "60", "70", "80", "100", "120"])
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