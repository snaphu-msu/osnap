from osnap import *
import sys

if __name__ == "__main__":
    
    run_date = "14may19"
    zams_mass = 20.0
    alpha = 1.25
    model_name = f"stir2_{run_date}_s{zams_mass}_alpha{alpha}"

    print(load_data.load_stitched_data(f"../data/nucleosynthesis/stitched_{model_name}"))
    #element = str(sys.argv[1])
    #plotting.plot_profile(stitched, element, save_path=f"plots/enclosed_mass_{element}.png", force_log = True)