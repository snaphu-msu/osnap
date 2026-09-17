

import os
from collections import defaultdict



SOURCE_FOLDER = "/Users/maruf/nucleo/CascadeProjects/windsurf-project-2/winnet/src"
TRAJECTORY_FOLDER = "/Users/maruf/Desktop/hermansenInput/Lagrangian_nu"
RUNS_FOLDER = os.path.join(SOURCE_FOLDER, "runs", "hermansen_runs_nuflag")
OUTPUT_FOLDER = os.path.join(SOURCE_FOLDER, "hermansen_workflow", "results_nuflag")

NUMBER_OF_TRACERS = 100


def read_enclosed_masses():
    """Read the enclosed mass represented by every tracer."""
    masses = []

    for tracer_number in range(NUMBER_OF_TRACERS):
        filename = (
            f"stir2_oct8_s12.0_alpha1.25_tracer{tracer_number}.dat"
        )
        filepath = os.path.join(TRAJECTORY_FOLDER, filename)

        with open(filepath, "r") as file:
            first_line = file.readline()

        # The header contains text such as: mass element = 1.2345
        mass_text = first_line.split("mass element =", 1)[1].split()[0]
        masses.append(float(mass_text))

    return masses


def calculate_zone_masses(enclosed_masses):
    """Give each tracer the mass halfway to its neighboring tracers."""
    zone_masses = []

    for tracer_number in range(NUMBER_OF_TRACERS):
        if tracer_number == 0:
            mass = 0.5 * (enclosed_masses[1] - enclosed_masses[0])
        elif tracer_number == NUMBER_OF_TRACERS - 1:
            mass = 0.5 * (
                enclosed_masses[-1] - enclosed_masses[-2]
            )
        else:
            mass = 0.5 * (
                enclosed_masses[tracer_number + 1]
                - enclosed_masses[tracer_number - 1]
            )

        zone_masses.append(mass)

    return zone_masses


def read_weighted_fractions(zone_masses):
    """Read Xi values and add Xi multiplied by the tracer's zone mass."""
    weighted_fractions = defaultdict(float)
    mass_from_files = 0.0

    for tracer_number in range(NUMBER_OF_TRACERS):
        run_name = f"tracer{tracer_number:04d}"
        filepath = os.path.join(RUNS_FOLDER, run_name, "finab.dat")

        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found; skipping this tracer.")
            continue

        tracer_mass = zone_masses[tracer_number]
        mass_from_files += tracer_mass

        with open(filepath, "r") as file:
            for line in file:
                if line.startswith("#"):
                    continue

                columns = line.split()
                if len(columns) < 5:
                    continue

                mass_number = int(columns[0])
                proton_number = int(columns[1])
                mass_fraction = float(columns[4])
                isotope = (mass_number, proton_number)

                weighted_fractions[isotope] += mass_fraction * tracer_mass

    return weighted_fractions, mass_from_files


def save_results(results):
    """Save the isotope averages as a readable text file and Excel file."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    text_path = os.path.join(
        OUTPUT_FOLDER, "final_mass_fractions_nuflag.txt"
    )
    with open(text_path, "w") as file:
        file.write("   A     Z          mean_X\n")
        for mass_number, proton_number, average in results:
            file.write(
                f"{mass_number:4d}  {proton_number:4d}  {average:14.6E}\n"
            )

    try:
        import openpyxl
    except ImportError:
        print(f"Saved text file: {text_path}")
        return

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Mass Fractions"
    sheet.append(["A", "Z", "mean_X"])

    for mass_number, proton_number, average in results:
        sheet.append([mass_number, proton_number, average])

    excel_path = os.path.join(
        OUTPUT_FOLDER, "final_mass_fractions_nuflag.xlsx"
    )
    workbook.save(excel_path)
    print(f"Saved text file: {text_path}")
    print(f"Saved Excel file: {excel_path}")


def main():
    enclosed_masses = read_enclosed_masses()
    zone_masses = calculate_zone_masses(enclosed_masses)
    weighted_fractions, used_mass = read_weighted_fractions(zone_masses)

    if used_mass == 0:
        raise RuntimeError("No finab.dat files were found.")

    results = []
    for (mass_number, proton_number), weighted_value in sorted(
        weighted_fractions.items()
    ):
        average = weighted_value / used_mass
        results.append((mass_number, proton_number, average))

    save_results(results)
    print(f"Total isotopes: {len(results)}")
    print(f"Used zone mass: {used_mass:.4f} M_sun")


if __name__ == "__main__":
    main()
