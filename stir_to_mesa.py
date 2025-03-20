import yt
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from progs import ProgModel
import sys
from scipy.interpolate import RegularGridInterpolator as rgi
import h5py
np.set_printoptions(threshold=sys.maxsize)

# Load configuration
stream = open('config.yaml', 'r')
configs = yaml.safe_load(stream)
progenitor_directory = configs['progenitor_directory']
model_template = configs['model_template']
runs_directory = configs['runs_directory']
pns_boundary_energy = configs['minimal_energy']
cell_edge_velocity = configs['cell_edge_velocity']
velocity_col_name = "v" if cell_edge_velocity else "u"
eos_file_path = configs['eos_file_path']

# Constants in CGS units
M_sun = 1.989E33 # Mass of the sun in grams
R_sun = 6.959E10 # Radius of the sun in centimeters
sigma_b = 5.669E-5 # Stefan-Boltzmann constant
G = 6.67430E-8 # Gravitational constant

nuclear_network = ['neut', 'h1', 'prot', 'he3', 'he4', 'c12', 'n14', 'o16', 'ne20', 'mg24', 'si28', 
                   's32', 'ar36', 'ca40', 'ti44', 'cr48', 'cr60', 'fe52', 'fe54', 'fe56', 'co56', 'ni56']



def shift_to_cell_edge(data):
    """
    Takes values which are cell centered and shifts them to the edge of the cell.

    Parameters:
        data : numpy array (e.g., mass_density)

    Returns:
        shifted_data : numpy array
    """

    shifted_data = np.zeros_like(data)
    shifted_data[:-1] = data[:-1] + 0.5 * (data[1:] - data[:-1])
    shifted_data[-1] = data[-1] + 0.5 * (data[-1] - data[-2])
    return shifted_data



class Star():
    def __init__(self, stir_output, prog):
        '''Combines STIR output (checkpoint/plot) data and progenitor data to allow exporting as MESA output..'''

        # Finds the first index in the progenitor data that is outside the STIR domain
        prog_data = prog['data']
        self.stir_domain_end = stir_output.shape[0]
        prog_domain = len(prog_data.loc[prog_data['enclosed_mass'].values > np.max(stir_output['enclosed_mass'].values)])

        # Combines STIR and progenitor data, interpolating progenitor values based by enclosed mass
        self.data = pd.DataFrame(index = pd.RangeIndex(stir_output.shape[0] + prog_domain), columns = stir_output.columns, dtype=float)
        for col in stir_output.columns:
            
            # If the column doesn't exist in progenitor data, notify us and skip it
            if not(col in prog_data):
                print(f"Column {col} missing from progenitor data.")
                continue
            
            # Fill in data for each domain
            self.data[col].values[:stir_output.shape[0]] = stir_output[col]
            self.data[col].values[stir_output.shape[0]:] = prog_data[col].values[-prog_domain:]

        # Calculate the total specific energy and total energy of each cell, including gravitational potential
        self.total_specific_energy = self.data['ener'].values + self.data['gpot'].values

        # Calculate the PNS mass cut using radial velocity
        # TODO: Check with Jared about whether MESA is going to use total specific energy or temperature
        #       If they're using the energy, we might need to use helmholtz to estimate energy instead.
        #       Or we might need to use one index earlier for the PNS masscut
        self.total_mass = np.sum(self.data['density'] * self.data['cell_volume'])
        self.pns_masscut_index = np.min(np.where(self.total_specific_energy >= 0)) - 1
        self.pns_masscut = self.data['enclosed_mass'].values[self.pns_masscut_index]
        self.pns_radius = self.data['r'].values[self.pns_masscut_index]

        # Find the mass of the star, and of the star outside the PNS
        self.xmstar = self.total_mass - self.pns_masscut * M_sun
        self.data["dq"] = self.data['density'] * self.data['cell_volume'] / self.xmstar

        # Stitches the progenitor data onto the STIR ouput
        self.plot_data(self.data['enclosed_mass'].values, ylog = True, xlabel = "Cell Index", ylabel = "log(enclosed mass)")
        self.plot_data(self.data['r'].values, ylog = True, xlabel = "Cell Index", ylabel = "log(radius)")
        self.plot_data(self.data['density'].values, ylog = True, xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass (M_sun)", ylabel = "log(mass density)")
        self.plot_data(self.data['pressure'].values, ylog = True, xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass (M_sun)", ylabel = "log(pressure)")
        self.plot_data(self.data['temp'].values, ylog = True, xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass (M_sun)", ylabel = "log(temp)")
        self.plot_data(self.data['cell_volume'], xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass (M_sun)", ylabel = "volume")
        self.plot_data(self.total_specific_energy, xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass (M_sun)", ylabel = "total specific energy")
        self.plot_data(self.data[velocity_col_name], xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass", ylabel = "radial velocity")

        # Excise the PNS from the data
        self.data = self.data.drop(np.arange(self.pns_masscut_index + 1))
        
        # Calculate the remaining energy of the star without the PNS
        self.total_energy = (self.data['ener'].values + self.data['gpot'].values) * self.data['density'].values * self.data['cell_volume'].values

        # Prepares the data for MESA output by adding missing columns
        # TODO: The second line here shouldn't be necessary once the kepler progenitor and STIR have these values calculated.
        self.data = self.data.assign(lnR = np.log(self.data['r'].values), lnd = np.log(self.data['density'].values), lnT = np.log(self.data['temp'].values))
        self.data = self.data.assign(mlt_vc = np.zeros(self.data.shape[0]))
        
        # MESA needs the surface luminosity which is the same as in progenitor, but all other values should just be a copy of that value
        self.data = self.data.assign(L = np.ones(self.data.shape[0]) * prog_data["L"].values[-1])

        self.prog_data = prog

    def plot_data(self, data, xaxis = None, xlabel = "", ylabel = "", xlog = False, ylog = False, zoom_width = 80):
        """
        Plots both the full star and a zoomed in region around the point at which STIR and the progenitor are stitched together.
        
        Parameters:
            self : Star object
            data : numpy array
                The data to be plotted.
            xaxis : numpy array
                The x-axis values for the data.
            xlabel : str
                The label for the x-axis.
            ylabel : str
                The label for the y-axis.
            xlog : bool
                Whether to plot the x-axis in log scale.
            ylog : bool
                Whether to plot the y-axis in log scale.
            zoom_width : int
                The full width of the zoomed in region around the stitch point.
        """

        # If no x-axis was specified, use the cell index
        if xaxis is None: xaxis = np.arange(len(data))
        used_xaxis = (np.log10(xaxis) if xlog else xaxis)

        # Plot the entire curve of stitched data
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot_data = np.log10(data) if ylog else data
        axs[0].plot(used_xaxis, plot_data)
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        axs[0].set_title(f"Full Star")
        axs[0].axvspan(used_xaxis[0], used_xaxis[self.pns_masscut_index], color="red", alpha=0.2, ec=None)
        axs[0].axvspan(used_xaxis[self.pns_masscut_index], used_xaxis[self.stir_domain_end], color="blue", alpha=0.2, ec=None)
        axs[0].axvspan(used_xaxis[self.stir_domain_end], used_xaxis[-1], color="green", alpha=0.2, ec=None)

        # Plot again but zoomed in on the stitch region
        zoom_left = max(0, self.stir_domain_end - zoom_width//2)
        zoom_right = min(len(data), self.stir_domain_end + zoom_width//2)
        axs[1].plot(used_xaxis[zoom_left : zoom_right], plot_data[zoom_left : zoom_right])
        axs[1].axvspan(used_xaxis[zoom_left], used_xaxis[self.stir_domain_end], color="blue", alpha=0.2, ec=None)
        axs[1].axvspan(used_xaxis[self.stir_domain_end], used_xaxis[zoom_right], color="green", alpha=0.2, ec=None)
        axs[1].set_title(f"Stitch Region")
        
        plt.show()



    def write_mesa_model(self):
        '''Writes the star's data into MESA input files.''' 
        
        # Easy way to convert values into the correct format for MESA model files
        def format_float(num):
            return f"{num:.16e}".replace('e', 'D')
        
        def format_int(num):
            return ' ' * (25 - int(np.floor(np.log10(num)))) + str(num)
        
        # Header Info
        # TODO: Is total energy including the PNS?
        avg_core_density = self.pns_masscut * M_sun / (4/3 * np.pi * self.pns_radius**3) 
        file_header = self.prog_data["header_start"] + f"""
                  version_number   {self.prog_data["version_number"]}
                          M/Msun      {format_float(self.total_mass / M_sun)}
                    model_number      {format_int(self.prog_data["model_number"])}
                        star_age      {format_float(self.prog_data["star_age"])}
                       initial_z      {format_float(self.prog_data["initial_z"])}
                        n_shells      {format_int(self.data.shape[0])}
                        net_name   {self.prog_data["net_name"]}
                         species      {format_int(self.prog_data["species"])}
                          xmstar      {format_float(self.xmstar)}  ! above core (g).  core mass: Msun, grams:      {format_float(self.pns_masscut)}    {format_float(self.pns_masscut * M_sun)}
                        R_center      {format_float(self.pns_radius)}  ! radius of core (cm).  R/Rsun, avg core density (g/cm^3):      {format_float(self.pns_radius / R_sun)}    {format_float(avg_core_density)}
                            Teff      {format_float(self.prog_data["Teff"])}
                  power_nuc_burn      {format_float(self.prog_data["power_nuc_burn"])}
                    power_h_burn      {format_float(self.prog_data["power_h_burn"])}
                   power_he_burn      {format_float(self.prog_data["power_he_burn"])}
                    power_z_burn      {format_float(self.prog_data["power_z_burn"])}
                     power_photo      {format_float(self.prog_data["power_photo"])}
                    total_energy      {format_float(np.sum(self.total_energy))}
         cumulative_energy_error      {format_float(self.prog_data["cumulative_energy_error"])}
   cumulative_error/total_energy      {format_float(self.prog_data["cumulative_error/total_energy"])}  log_rel_run_E_err      {format_float(self.prog_data["log_rel_run_E_err"])}
                     num_retries                               0

                lnd                        lnT                        lnR                          L                         dq                          {velocity_col_name}                     mlt_vc                   neut                       h1                         prot                       he3                        he4                        c12                        n14                        o16                        ne20                       mg24                       si28                       s32                        ar36                       ca40                       ti44                       cr48                       cr60                       fe52                       fe54                       fe56                       co56                       ni56     
"""

        # Data that will be written in table form in the stir_output.mod file
        # Also reverses the order of rows in the table so that the first cell is the outer radius and last cell is the center
        mesa_columns = np.concat((['lnd', 'lnT', 'lnR', 'L', 'dq', velocity_col_name, 'mlt_vc'], nuclear_network))
        mesa_input = self.data[mesa_columns].iloc[::-1].reset_index(drop=True)

        # Add one line for each cell, consisting of all it's properties
        new_lines = []
        for line_index in range(self.data.shape[0]):

            # Writes the cell/line index 
            spaces = 4 - int(np.floor(np.log10(line_index + 1)))
            new_line = ' ' * spaces + str(line_index + 1)

            # Writes each of the properties
            for column_name in mesa_input.columns:
                spaces = 5 if mesa_input.at[line_index, column_name] >= 0 else 4
                new_line += spaces * ' ' + format_float(mesa_input.at[line_index, column_name])

            new_lines.append(new_line)

        # Footer containing info about the previous model
        file_footer = f"""
        
        previous model

               previous n_shells      {format_int(self.prog_data['data'].shape[0])}
           previous mass (grams)      {format_float(self.prog_data["M/Msun"] * M_sun)}
              timestep (seconds)      {format_float(self.prog_data["timestep"])} 
               dt_next (seconds)      {format_float(self.prog_data["dt_next"])}"""

        # Write all of the above to the stir_output.mod file
        with open(f'stir_output.mod', 'w') as file:
            file.writelines(file_header + '\n'.join(new_lines) + file_footer)
            print(f"Successfully created/updated 'stir_output.mod'")



# TODO: Missing prot (maybe same as h1), cr60, co56
# TODO: May need to convert velocity to the cell center riemann velocity if using not using cell edge
# TODO: Will produce errors due to changes in data structure and needed variables.
def read_kepler_progenitor(progenitor_mass):
    '''
    Reads in kepler progenitor data, using the progs package, and returns a dataframe.
    Also ensures the returned data has the same column names as the STIR output for compatibility purposes.
    '''
    prog_data = ProgModel(str(progenitor_mass), "sukhbold_2016").profile
    prog_data.rename(columns={"radius_edge": "r", "temperature": "temp", "neutrons": "neut", "luminosity": "L", "energy": "ener"}, inplace=True)
    #print(prog_data.columns.to_list())

    # TODO: Temporary measure until I figure out what to do about missing compositions
    prog_data = prog_data.assign(prot = np.zeros(prog_data.shape[0]), co56 = np.zeros(prog_data.shape[0]), cr60 = np.zeros(prog_data.shape[0]))

    # Calculates the progenitor volume, enclosed mass, and gravitational potential
    prog_volume = (4/3) * np.pi * np.diff(np.concatenate(([0], prog_data['r'].values**3)))
    prog_data = prog_data.assign(cell_volume = prog_volume) 
    prog_data = prog_data.assign(enclosed_mass = np.cumsum(prog_data['cell_volume'].values * prog_data['density'].values) / M_sun)
    prog_data = prog_data.assign(gpot = -G * prog_data['enclosed_mass'].values / prog_data['r'].values)

    return prog_data



def read_mesa_progenitor(profile_path, model_path):
    """
    Reads the output data from a MESA model and stores the necessary variables.
    Also ensures the returned data has the same column names as the STIR output for compatibility purposes.
    """
    
    prog_data = {}
    with open(profile_path, "r") as file:
        lines = file.readlines()

        # Find the columns that contain the necessary data
        # WARNING: There is a small chance of conv_vel not being exactly mlt_vc depending on the time step.
        needed_columns = np.concat((['mass', 'logRho', 'temperature', 'radius_cm', 'luminosity', 
                                     'logdq', 'velocity', 'conv_vel', 'energy', 'pressure'], nuclear_network))
        input_column_names = lines[5].split()
        column_indices = [input_column_names.index(col) for col in needed_columns if col in input_column_names]

        # Process the numerical data into a 2D array
        structured_data = [list(map(float, np.array(line.split())[column_indices])) for line in lines[6:]]

        # If any columns are missing from the progenitor, fill them with a very small number. Generally only occurs with composition.
        for i, col in enumerate(needed_columns):
            if col not in input_column_names:
                print(f"Column {col} missing from progenitor data. Filling with very small number for all cells.")
                for j in range(len(structured_data)): 
                    structured_data[j].insert(i, 1e-99)

        # Read the numerical data into a pandas dataframe, renaming columns for compatibility and ease of use
        output_column_names = np.concat((['enclosed_mass', 'density', 'temp', 'r', 'L', 'dq', velocity_col_name, 
                                          'mlt_vc', 'ener', 'pressure'], nuclear_network))
        prog_data["data"] = pd.DataFrame(structured_data, columns=output_column_names).iloc[::-1].reset_index(drop=True)

        # Convert data to the correct scale for compatibility with STIR output
        prog_data["data"]["density"] = (10 ** prog_data["data"]['density'])
        prog_data["data"]["dq"] = 10 ** prog_data["data"]['dq']

        # Calculates the progenitor volume and gravitational potential
        prog_volume = (4/3) * np.pi * np.diff(np.concatenate(([0], prog_data["data"]['r'].values**3)))
        prog_data["data"] = prog_data["data"].assign(cell_volume = prog_volume) 
        prog_data["data"] = prog_data["data"].assign(gpot = -G * prog_data["data"]['enclosed_mass'].values / prog_data["data"]['r'].values)

    # Reads from the .mod file to get necessary header and footer information
    with open(model_path, "r") as file:

        lines = file.readlines()
        prog_data["header_start"] = lines[0] + lines[1] + lines[2]
        for i in np.concat((range(4, 20), range(-3, -1))):
            line_data = lines[i].split()
            value = line_data[-1]
            if "D+" in value or "D-" in value:
                prog_data[line_data[0]] = float(value.replace("D", "e"))
            elif value.isdigit():
                prog_data[line_data[0]] = int(value)
            else:
                prog_data[line_data[0]] = value

        # These are on the same line so they have to be handled separately
        prog_data["cumulative_error/total_energy"] = float(lines[20].split()[1].replace("D", "e"))
        prog_data["log_rel_run_E_err"] = float(lines[20].split()[3].replace("D", "e"))
        

    return prog_data



def read_stir_output(path):
    '''Reads in data from a STIR checkpoint (or plot) file, making modifications and returning it as a dataframe.'''

    # Load the equation of state used by STIR
    EOS = h5py.File(eos_file_path,'r')
    mif_logenergy = rgi((EOS['ye'], EOS['logtemp'], EOS['logrho']), EOS['logenergy'][:,:,:], bounds_error=False)
    
    # Load the STIR checking/plot data into a dataframe
    stir_data = yt.load(path).all_data()
    grab_data = [("gas", "density"), ("flash", "temp"), ("gas", "r"), ("flash", "velx"), ("gas", "pressure"),# ("flash", "ye  "),
                 ("flash", "cell_volume"), ("flash", "ener"), ("flash", "gpot")]
    stir_dataframe = stir_data.to_dataframe(grab_data)
    
    # Shift radius to cell edge for compatibility with MESA radii
    stir_dataframe['r'] = shift_to_cell_edge(stir_dataframe['r'].values) # Shifts radius to the cell edge to match progenitor outputs
    
    # Calculate the enclosed mass for every zone
    stir_dataframe = stir_dataframe.assign(enclosed_mass = np.cumsum(stir_dataframe['cell_volume'].values * stir_dataframe['density'].values) / M_sun)

    # Rename the velocity column to match MESA output
    stir_dataframe.rename(columns={"velx": velocity_col_name}, inplace=True)

    # Use the STIR EOS to calculate the total specific energy
    lye = stir_data['ye  ']
    llogtemp = np.log10(stir_data['temp'] * 8.61733326e-11)
    llogrho = np.log10(stir_data['dens'])
    energy = 10.0 ** mif_logenergy(np.array([lye, llogtemp, llogrho]).T)
    llogtemp = llogtemp * 0.0 - 2.0
    energy0 = 10.0 ** mif_logenergy(np.array([lye, llogtemp, llogrho]).T)
    stir_dataframe["ener"] = (0.5 * stir_data['velx'] ** 2 + (energy - energy0) * yt.units.erg / yt.units.g).v

    # If using cell edge velocity, shift the STIR velocities to the cell edge
    if cell_edge_velocity: 
        stir_dataframe[velocity_col_name] = shift_to_cell_edge(stir_dataframe[velocity_col_name].values)

    # Add nuclear network with all zeros for mass fractions
    for col in nuclear_network:
        stir_dataframe[col] = np.zeros_like(stir_dataframe.shape[0])

    return stir_dataframe