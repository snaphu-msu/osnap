import yt
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from progs import ProgModel
import sys
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
    def __init__(self, stir_data, prog_data):
        '''Combines STIR output (checkpoint) data and progenitor data to allow exporting as MESA output..'''

        # Finds the first index in the progenitor data that is outside the STIR domain
        prog_domain = len(prog_data.loc[prog_data['enclosed_mass'].values > np.max(stir_data['enclosed_mass'].values)])
        self.stitch_index = prog_data.shape[0] - prog_domain

        if prog_domain == 0:
            print("Progenitor data is entirely within STIR domain. No stitching necessary.")

        # Combines STIR and progenitor data, interpolating STIR values based by enclosed mass
        self.data = prog_data.copy()
        for col in stir_data.columns:
            if col == "enclosed_mass": continue
            
            # And If the column doesn't exist in progenitor data, notify us and skip it
            if not(col in self.data):
                print(f"Column {col} missing from progenitor data.")
                continue

            # If this data is composition, just overwrite with the progenitor values since STIR doesn't simulate the nuclear network
            # TODO: This is temporary, as composition should definitely change during the STIR simulation. STIR2 does have a nuclear network but is not quite ready.
            if col in nuclear_network:
                self.data.loc[0:self.stitch_index - 1, col] = prog_data['enclosed_mass'].values[:self.stitch_index]
                continue

            # Otherwise, regrid the STIR data to fit the progenitor mass coordinates
            interp_data = np.interp(prog_data['enclosed_mass'].values[:self.stitch_index], stir_data['enclosed_mass'].values, stir_data[col])
            self.data.loc[0:self.stitch_index - 1, col] = interp_data

        # Calculate the total specific energy and total energy of each cell, including gravitational potential
        self.total_specific_energy = self.data['ener'].values - 1.275452534480232E+018 + self.data['gpot'].values
        self.total_energy = (self.data['ener'].values - 1.275452534480232E+018 + self.data['gpot'].values) * self.data['density'].values  * self.data['cell_volume'].values

        # Stitches the progenitor data onto the STIR ouput
        self.plot_data(self.data['enclosed_mass'].values, ylog = True, xlabel = "Cell Index", ylabel = "log(enclosed mass)")
        self.plot_data(self.data['r'].values, ylog = True, xlabel = "Cell Index", ylabel = "log(radius)")
        self.plot_data(self.data['density'].values, ylog = True, xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass (M_sun)", ylabel = "log(mass density)")
        self.plot_data(self.data['pressure'].values, ylog = True, xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass (M_sun)", ylabel = "log(pressure)")
        self.plot_data(self.data['temp'].values, ylog = True, xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass (M_sun)", ylabel = "log(temp)")
        self.plot_data(self.data['cell_volume'], xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass (M_sun)", ylabel = "volume")
        self.plot_data(self.total_specific_energy, xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass (M_sun)", ylabel = "total specific energy")
        self.plot_data(self.total_energy, xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass (M_sun)", ylabel = "total energy")
        self.plot_data(self.data[velocity_col_name], xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass", ylabel = "radial velocity")

        self.total_mass = np.sum(self.data['density'] * self.data['cell_volume'])

        # Calculate the PNS mass cut using radial velocity
        masscut_index = np.min(np.where(self.total_specific_energy >= 0))
        self.pns_masscut = self.data['enclosed_mass'].values[masscut_index]

        # Prepares the data for MESA output by adding missing columns
        # TODO: The second line here shouldn't be necessary once the kepler progenitor and STIR have these values calculated.
        self.data = self.data.assign(lnR = np.log(self.data['r'].values), lnd = np.log(self.data['density'].values), lnT = np.log(self.data['temp'].values))
        self.data = self.data.assign(L = np.zeros(self.data.shape[0]), dq = np.zeros(self.data.shape[0]), u = np.zeros(self.data.shape[0]), mlt_vc = np.zeros(self.data.shape[0]))

    
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
        stitch_left = used_xaxis[self.stitch_index - 1]
        stitch_right = used_xaxis[min(self.stitch_index, len(used_xaxis) - 1)]

        # Plot the entire curve of stitched data
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot_data = np.log10(data) if ylog else data
        axs[0].plot(used_xaxis, plot_data)
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        axs[0].set_title(f"Full Star")
        axs[0].axvspan(stitch_left, stitch_right, color="red", alpha=0.3)

        # Plot again but zoomed in on the stitch region
        zoom_left = max(0, self.stitch_index - zoom_width//2)
        zoom_right = min(len(data), self.stitch_index + zoom_width//2)
        axs[1].plot(used_xaxis[zoom_left : zoom_right], plot_data[zoom_left : zoom_right])
        axs[1].axvspan(stitch_left, stitch_right, color="red", alpha=0.3)
        axs[1].set_title(f"Stitch Region")
        
        plt.show()



    def write_mesa_model(self, model_index):
        '''Writes the star's data into MESA input files.''' 
        
        # Easy way to convert values into the correct format for MESA model files
        def format_number(number):
            return f"{number:.16e}".replace('e', 'D')
        
        n_shells = ' ' * (25 - int(np.floor(np.log10(self.data.shape[0])))) + str(self.data.shape[0])
        total_energy = np.sum(self.total_energy) 
        model_number = ' ' * (25 - int(np.floor(np.log10(model_index)))) + str(model_index)
        star_age = 6.3376175628057906E-09 # TODO: Do we need this?
        initial_z = 2.0000000000000000E-02 # TODO: Do we need this?
        R_center = 3.9992663820663236E+07 # TODO: Radius of PNS mass cut
        xmstar = self.total_mass - self.pns_masscut # Mass of star minus PNS mass cut
        avg_core_density = self.pns_masscut / (4/3 * np.pi * R_center**3) 
        Teff = 7.2257665281116360E+03 # TODO: Use the stefan-boltzmann law after luminosity is calculated

        # TODO: Verify none of the values that are zeroed out below aren't necessary
        # TODO: May need to specify things like version number, nuclear network name, species count, etc either from progenitor or from config file
        file_header = f"""! note: initial lines of file can contain comments
!
           548 -- model for mesa/star, cell center Riemann velocities (u), mlt convection velocity (mlt_vc). cgs units. lnd=ln(density), lnT=ln(temperature), lnR=ln(radius), L=luminosity, dq=fraction of xmstar=(mstar-mcenter) in cell; remaining cols are mass fractions.

                  version_number   'r24.08.1'
                          M/Msun      {format_number(self.total_mass / M_sun )}
                    model_number      {model_number}
                        star_age      {format_number(star_age)}
                       initial_z      {format_number(initial_z)}
                        n_shells      {n_shells}
                        net_name   'approx21_cr60_plus_co56.net'
                         species                              22
                          xmstar      {format_number(xmstar)}  ! above core (g).  core mass: Msun, grams:      {format_number(self.pns_masscut / M_sun)}    {format_number(self.pns_masscut)}
                        R_center      {format_number(R_center)}  ! radius of core (cm).  R/Rsun, avg core density (g/cm^3):      {format_number(R_center / R_sun)}    {format_number(avg_core_density)}
                            Teff      {format_number(Teff)}
                  power_nuc_burn      0.0000000000000000D+00
                    power_h_burn      0.0000000000000000D+00
                   power_he_burn      0.0000000000000000D+00
                    power_z_burn      0.0000000000000000D+00
                     power_photo      0.0000000000000000D+00
                    total_energy      {format_number(total_energy)}
         cumulative_energy_error      0.0000000000000000D+00
   cumulative_error/total_energy      0.0000000000000000D+00  log_rel_run_E_err      0.0000000000000000D+00
                     num_retries                               0

                lnd                        lnT                        lnR                          L                         dq                          {velocity_col_name}                     mlt_vc                   neut                       h1                         prot                       he3                        he4                        c12                        n14                        o16                        ne20                       mg24                       si28                       s32                        ar36                       ca40                       ti44                       cr48                       cr60                       fe52                       fe54                       fe56                       co56                       ni56     
"""

        # Data that will be written in table form in the stir_output.mod file
        # Also reverses the order of rows in the table so that the first cell is the outer radius and last cell is the center
        mesa_columns = np.concat((['lnd', 'lnT', 'lnR', 'L', 'dq', velocity_col_name, 'mlt_vc'], nuclear_network))
        if cell_edge_velocity: mesa_columns[5] = "v"
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
                new_line += spaces * ' ' + format_number(mesa_input.at[line_index, column_name])

            new_lines.append(new_line)

        # Footer containing info about the previous model
        # TODO: Currently these values are fake. Do we need to set them? Should n_shells be the number of shells in the progenitor?
        file_footer = f"""
        
        previous model

               previous n_shells      {n_shells}
           previous mass (grams)      2.3377068540442852D+34 
              timestep (seconds)      4.2684831294390047D-04 
               dt_next (seconds)      5.1221797553268049D-04"""

        # Write all of the above to the stir_output.mod file
        with open(f'model_{model_index}/stir_output.mod', 'w') as file:
            file.writelines(file_header + '\n'.join(new_lines) + file_footer)
            print(f"Successfully created/updated 'model_{model_index}/stir_output.mod'")


# TODO: Need to calculate dq (xmstar=(mstar-mcenter)), u (cc riemann velocity), and mlt_vc
# TODO: Missing prot (maybe same as h1), cr60, co56
# TODO: May need to convert velocity to the cell center riemann velocity
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



def read_mesa_progenitor(path):
    """
    Reads the output data from a MESA model and stores the necessary variables.
    Also ensures the returned data has the same column names as the STIR output for compatibility purposes.
    """
    
    with open(path, "r") as file:
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
        prog_data = pd.DataFrame(structured_data, columns=output_column_names).iloc[::-1].reset_index(drop=True)

        # Convert data to the correct scale for compatibility with STIR output
        prog_data["density"] = (10 ** prog_data['density'])
        prog_data["dq"] = 10 ** prog_data['dq']

        # Calculates the progenitor volume and gravitational potential
        prog_volume = (4/3) * np.pi * np.diff(np.concatenate(([0], prog_data['r'].values**3)))
        prog_data = prog_data.assign(cell_volume = prog_volume) 
        prog_data = prog_data.assign(gpot = -G * prog_data['enclosed_mass'].values / prog_data['r'].values)

        print(np.sum(prog_data['dq'].values))

        return prog_data



# TODO: Need to calculate L, dq (xmstar=(mstar-mcenter)), u (cc riemann velocity), and mlt_vc
def read_stir_checkpoint(path):
    '''Reads in data from a STIR checkpoint file, making modifications and returning it as a dataframe.'''
    stir_data = yt.load(path).all_data()
    grab_data = [("gas", "density"), ("flash", "temp"), ("gas", "r"), ("flash", "velx"), ("gas", "pressure"),# ("flash", "ye  "),
                 ("flash", "cell_volume"), ("flash", "ener"), ("flash", "gpot")]
    stir_dataframe = stir_data.to_dataframe(grab_data)
    stir_dataframe['r'] = shift_to_cell_edge(stir_dataframe['r'].values) # Shifts radius to the cell edge to match progenitor outputs
    stir_dataframe = stir_dataframe.assign(enclosed_mass = np.cumsum(stir_dataframe['cell_volume'].values * stir_dataframe['density'].values) / M_sun)

    # Rename the velocity column to match MESA output
    stir_dataframe.rename(columns={"velx": velocity_col_name}, inplace=True)

    # If using cell edge velocity, shift the STIR velocities to the cell edge
    if cell_edge_velocity: 
        stir_dataframe[velocity_col_name] = shift_to_cell_edge(stir_dataframe[velocity_col_name].values) # Shifts radial velocity to the cell edge to match progenitor outputs

    # Add nuclear network with all zeros for mass fractions
    for col in nuclear_network:
        stir_dataframe[col] = stir_dataframe.shape[0]

    return stir_dataframe