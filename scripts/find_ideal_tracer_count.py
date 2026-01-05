from osnap import *
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_total_mass(data, isotope):
    return np.sum(data[isotope] * data["density"] * data["cell_volume"]) / 1.98e33

if __name__ == "__main__":

    base_path = config.stitched_output_directory + "/"
    run_date = "14may19"
    zams_mass = 20.0
    alpha = 1.25
    tracer_counts = [
        40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 
        550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, 
        1600, 1700, 1800, 1900, 2000, 2200, 2400, 2500
    ]
    
    # Establish the ground truth as the abundances for the max tracer count model
    ground_truth_data = pd.read_csv(base_path + f"stitched_stir2_{run_date}_s{zams_mass}_alpha{alpha}_n{tracer_counts[-1]}", sep='\s+')
    isotopes = ground_truth_data.columns.values[11:]
    ground_truth = {isotope: get_total_mass(ground_truth_data, isotope) for isotope in isotopes}

    # Initialize dictionaries that will hold the relative and absolute errors for each isotope
    relative_errors = {isotope: [] for isotope in isotopes}
    absolute_errors = {isotope: [] for isotope in isotopes}

    # Calculate the errors for each isotope at each tracer count
    for tracers in tracer_counts:
        model_name = f"stitched_stir2_{run_date}_s{zams_mass}_alpha{alpha}_n{tracers}"
        data = pd.read_csv(base_path + model_name, sep='\s+')
        for isotope in isotopes:
            absolute_errors[isotope].append(np.abs(ground_truth[isotope] - get_total_mass(data, isotope)))
            relative_errors[isotope].append(absolute_errors[isotope][-1] / ground_truth[isotope])

    # Find the mean errors for each isotope across all tracer counts
    mean_relative_error = pd.DataFrame(relative_errors).mean(axis=1)
    mean_absolute_error = pd.DataFrame(absolute_errors).mean(axis=1)

    # Plot both the relative and absolute errors vs tracer count for a variety of isotopes
    plt.figure(figsize=(20, 12))
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Error vs Tracer Count")

    axs[0].plot(tracer_counts, mean_relative_error, linestyle='--', marker='.', label='mean')
    axs[0].plot(tracer_counts, relative_errors["he4"], label="he4", marker='.')
    axs[0].plot(tracer_counts, relative_errors["fe56"], label="fe56", marker='.')
    axs[0].plot(tracer_counts, relative_errors["ti44"], label="ti44", marker='.')
    axs[0].plot(tracer_counts, relative_errors["ni56"], label="ni56", marker='.')
    axs[0].plot(tracer_counts, relative_errors["ni58"], label="ni58", marker='.')
    axs[0].plot(tracer_counts, relative_errors["ni64"], label="ni64", marker='.')
    axs[0].loglog()
    axs[0].set_ylabel("Relative Error")

    axs[1].plot(tracer_counts, mean_absolute_error, linestyle='--', marker='.', label='mean')
    axs[1].plot(tracer_counts, absolute_errors["he4"], label="he4", marker='.')
    axs[1].plot(tracer_counts, absolute_errors["fe56"], label="fe56", marker='.')
    axs[1].plot(tracer_counts, absolute_errors["ti44"], label="ti44", marker='.')
    axs[1].plot(tracer_counts, absolute_errors["ni56"], label="ni56", marker='.')
    axs[1].plot(tracer_counts, absolute_errors["ni58"], label="ni58", marker='.')
    axs[1].plot(tracer_counts, absolute_errors["ni64"], label="ni64", marker='.')
    axs[1].loglog()
    axs[1].set_xlabel("Tracer Count")
    axs[1].set_ylabel("Absolute Error")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.plot_directory + "/tracer_count_analysis.png")
