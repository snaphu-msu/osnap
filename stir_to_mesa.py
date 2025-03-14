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

# Constants in CGS units
M_sun = 1.989E33 # Mass of the sun in grams
R_sun = 6.959E10 # Radius of the sun in centimeters
sigma_b = 5.669E-5 # Stefan-Boltzmann constant

# Other constants
pns_boundary_energy = 2e18 # Energy at which the PNS boundary is defined
# TODO: This is pretty close to the number used in energy calculatations below...



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



def stitch_regions(stir_data, prog_data, prog_start, smoothing = 50, xaxis = None, plot = True, xlabel = "", ylabel = "", xlog = True, ylog = True):
    """
    Stitches the progenitor data onto the end of the STIR domain, smoothing out the stitched region and plotting for inspection.

    Parameters:
        stir_data : numpy array (e.g., STIR mass_density)
        prog_data : numpy array (e.g., Progenitor mass_density)
        prog_start : integer (first index of progenitor data is outside of STIR domain)
        smoothing : integer (WIP currently meaningless, except that when 0 no smoothing is applied)
        xaxis : numpy array (the data to use as our x-values both when predicting data points and for plotting, usually cell index, log(radius), or log(enclosed mass))
        plot : boolean (whether or not to show the plots for inspection)
        xlabel : string (label for the plot's x-axis)
        ylabel : string (label for the plot's y-axis)
        xlog : boolean (whether not to put the xaxis variable into log-space)
        ylog : boolean (whether not to put the yaxis variable into log-space)

    Returns:
        stitched_data : numpy array
    """


    smoothing = 0

    # Stitch the two datasets together
    stitched_data = np.concat((stir_data, prog_data[prog_start:]))

    # If no x-axis was specified, use the cell index
    if xaxis is None: xaxis = np.arange(len(stitched_data))
    used_xaxis = np.log10(xaxis) if xlog else xaxis

    # Smooths out the connection between the stir domain and progenitor data outside that domain
    # Since STIR has evolved past the progenitor, there is otherwise a sudden increase in most variables at the stitch
    # TODO: Still needs to actually use the predicted values up to the intersection
    if smoothing > 0:

        # Predict what the values for stir data would be outside the stir domain
        pred_cells = 60   # How far behind do we want to use for the linear fit?
        pred_sample = np.log10(stir_data) if ylog else stir_data
        pred_coeffs = np.polyfit(used_xaxis[len(pred_sample) - pred_cells : len(pred_sample)], pred_sample[-pred_cells:], 1)
        pred_values = np.poly1d(pred_coeffs)(used_xaxis[len(pred_sample) :])
        
        # If predicting used log-space, then convert predictions back to the original scale
        if ylog: pred_values = 10 ** pred_values

        # Find the index at which the predicted values intersect with the real values from the progenitor
        intersection = np.where(np.diff(np.sign(pred_values[:] - stitched_data[len(stir_data):])))[0][0]

    # If not smoothing, set the intersection to some value since it determines the zoom width in the second plot
    else: 
        intersection = 30

    if plot:

        # Plot the entire curve of stitched data
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot_data = np.log10(stitched_data) if ylog else stitched_data
        axs[0].plot(used_xaxis, plot_data)
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        axs[0].set_title(f"Full Star")
        axs[0].axvline(used_xaxis[len(stir_data)], ls="--", c="red")

        # Plot again but zoomed in on the stitch region
        zoom_left_edge = max(0, len(stir_data) - 100)
        zoom_right_edge = min(len(stitched_data), len(stir_data) + intersection + 10)
        axs[1].set_xlim(used_xaxis[zoom_left_edge], used_xaxis[zoom_right_edge])
        axs[1].set_ylim(np.min(plot_data[zoom_left_edge:zoom_right_edge]), np.max(plot_data[zoom_left_edge:zoom_right_edge]))
        axs[1].plot(used_xaxis, plot_data)
        axs[1].axvline(used_xaxis[len(stir_data)], ls="--", c="red")
        axs[1].set_title(f"Stitch Region")

        # If the stitch region is smoothed, show a line of predicted values on both plots
        if smoothing > 0:
            predicted_plot_data = np.log10(pred_values) if ylog else pred_values
            axs[0].plot(used_xaxis[len(stir_data) : len(stir_data) + len(predicted_plot_data)], predicted_plot_data, ls="--")
            axs[0].axvline(used_xaxis[len(stir_data) + intersection], ls="--", c="green")
            axs[1].plot(used_xaxis[len(stir_data) : zoom_right_edge], predicted_plot_data[:zoom_right_edge - len(stir_data)], ls="--")
            axs[1].axvline(used_xaxis[len(stir_data) + intersection], ls="--", c="green")
        
        plt.show()

    return stitched_data



def calculate_mass_coordinates(radius, mass_density):
    '''Calculates the mass coordinate for each cell in the star using the mass density and radius.'''
    r_avg = 0.5 * (radius[:-1] + radius[1:])
    dr = radius[1:] - radius[:-1]
    dm = 4 * np.pi * r_avg**2 * mass_density[:-1] * dr
    mass_coordinate = np.zeros(len(radius))
    mass_coordinate[1:] = np.cumsum(dm)
    return mass_coordinate



class Star():
    def __init__(self, stir_data, prog_data):
        '''Reads data from a STIR checkpoint at the path and stores the necessary variables.'''

        # Calculates the progenitor volume
        prog_volume = np.zeros(prog_data.shape[0])
        prog_volume[0] = (4/3) * np.pi * prog_data['r'].values[0] ** 3
        prog_volume[1:] = (4/3) * np.pi * (prog_data['r'].values[1:] ** 3 - prog_data['r'].values[:-1] ** 3)

        # Shifts radius values to the cell edge to match MESA output
        self.radius = shift_to_cell_edge(stir_data['r'].values)

        # Convert to lagrangian coordinate system
        # TODO: Currently only gets mass coordinates but doesn't adjust any values of existing variables
        self.enclosed_mass = calculate_mass_coordinates(self.radius, stir_data['density'].values)
        prog_enclosed_mass = calculate_mass_coordinates(prog_data['r'].values, prog_data['density'].values)

        # Finds the first index in the progenitor data that is outside the STIR domain
        prog_domain = len(prog_data.loc[prog_data['r'].values > np.max(self.radius)])
        stitch_index = prog_data.shape[0] - prog_domain
        self.cell_count = len(self.radius) + prog_domain

        # TODO: MESA says it uses edge velocity, but MESA model files need cell-center riemann velocities?
        self.radial_velocity = shift_to_cell_edge(stir_data['velx'].values)

        # Calculate the total specific energy, including gravitational potential
        self.total_specific_energy = stir_data['ener'].values - 1.275452534480232E+018 + stir_data['gpot'].values
        prog_total_specific_energy = prog_data['ener'].values - 1.275452534480232E+018 + prog_data['gpot'].values

        # Stitches the progenitor data onto the STIR ouput
        self.enclosed_mass = stitch_regions(self.enclosed_mass, prog_enclosed_mass, stitch_index, smoothing = 0, xlog = False, xlabel = "Cell Index", ylabel = "log(enclosed mass)")
        self.radius = stitch_regions(self.radius, prog_data['r'].values, stitch_index, smoothing = 0, xlog = False, xlabel = "Cell Index", ylabel = "log(radius)")
        self.mass_density = stitch_regions(stir_data['density'].values, prog_data['density'].values, stitch_index, xaxis = self.enclosed_mass, xlabel = "log(enclosed mass)", ylabel = "log(mass density)")
        self.pressure = stitch_regions(stir_data['pressure'].values, prog_data['pressure'].values, stitch_index, xaxis = self.enclosed_mass, xlabel = "log(enclosed mass)", ylabel = "log(pressure)")
        self.temp = stitch_regions(stir_data['temp'].values, prog_data['temp'].values, stitch_index, xaxis = self.enclosed_mass, xlabel = "log(enclosed mass)", ylabel = "log(temp)")
        self.radial_velocity = stitch_regions(self.radial_velocity, prog_data['velx'].values, stitch_index, xlog = False, ylog = False, xaxis = self.enclosed_mass, xlabel = "log(enclosed mass)", ylabel = "radial velocity")
        self.volume  = np.concat((stir_data['cell_volume'].values, prog_volume))
        #self.total_specific_energy = stitch_regions(self.total_specific_energy, prog_total_specific_energy, stitch_index, ylog = False, xaxis = self.enclosed_mass, xlabel = "log(mass)", ylabel = "total specific energy")

        # Calculating the total energy of each cell in the star
        #self.total_energy = (stir_data['ener'] - 1.275452534480232E+018 + stir_data['gpot']) * stir_data['density'].v  * stir_data['cell_volume']

        # DEBUG: Plot the total specific energy with respect to log(radius) to find what a "small amount of energy" might be
        plt.plot(np.log10(self.radius[:len(stir_data['density'].values)]), self.total_specific_energy)
        plt.axhline(0, ls = "--", c = "red")
        plt.axhline(pns_boundary_energy, ls = "--", c = "green")
        plt.ylim(-0.05e20, 0.15e20)

        # Estimate the PNS boundary as the point where the total specific energy is first above some small energy (such as 2e18)
        pns_index = np.min(np.where(self.total_specific_energy < pns_boundary_energy))

        # Calculate the overburden energy as the total energy of all cells past the estimated PNS boundary
        # TODO: This needs to include the progenitor domain as well but we need to calcualte gpot to get total specific energy for the progenitors
        overburden_energy = np.sum(self.total_specific_energy[pns_index:])

        # Go back to the initial PNS index and integrate the specific total energy with a changing mass until the sum is equal to the overburden energy.
        # The index at that point is where the PNS star should stop.
        
        self.pns_masscut = np.sum(self.mass_density[:pns_index] * self.volume[:pns_index])
        
        # Structuring the data for each writing of MESA model files
        self.data = {}
        self.data['lnR'] = np.log(self.radius)
        self.data['lnd'] = np.log(self.mass_density)
        self.data['lnT'] = np.log(self.temp)

        # TODO: Values that need to be calculated somehow
        self.data['L'] = np.zeros(self.cell_count) # Luminosity TODO: Should I calculate this using the total energy and some dynamical time?
        self.data['dq'] = np.zeros(self.cell_count) # TODO: Fraction of xmstar=(mstar-mcenter) - what does this mean?
        self.data['u'] = np.zeros(self.cell_count) # TODO: Cell center riemann velocity - Is this simply the velocity of each edge averaged?
        self.data['mlt_vc'] = np.zeros(self.cell_count) # TODO: MLT convection velocity - How can I calculate this?

        # TODO: Nuclear Network (will need to interpolate values for the STIR domain for now)
        # TODO: Keep in mind that STIR does not simulate the nuclear network so the values within the STIR domain will be inaccurate
        self.data['neut'] = np.zeros(self.cell_count) # neutrons in progenitor data
        self.data['h1'] = np.zeros(self.cell_count)
        self.data['prot'] = np.zeros(self.cell_count) # not tracked separately from h1 in progenitor data
        self.data['he3'] = np.zeros(self.cell_count)
        self.data['he4'] = np.zeros(self.cell_count)
        self.data['c12'] = np.zeros(self.cell_count)
        self.data['n14'] = np.zeros(self.cell_count)
        self.data['o16'] = np.zeros(self.cell_count)
        self.data['ne20'] = np.zeros(self.cell_count) 
        self.data['mg24'] = np.zeros(self.cell_count)
        self.data['si28'] = np.zeros(self.cell_count) 
        self.data['s32'] = np.zeros(self.cell_count)
        self.data['ar36'] = np.zeros(self.cell_count) 
        self.data['ca40'] = np.zeros(self.cell_count)
        self.data['ti44'] = np.zeros(self.cell_count)
        self.data['cr48'] = np.zeros(self.cell_count)
        self.data['cr60'] = np.zeros(self.cell_count) # Missing in progenitor data?
        self.data['fe52'] = np.zeros(self.cell_count) 
        self.data['fe54'] = np.zeros(self.cell_count)
        self.data['fe56'] = np.zeros(self.cell_count) # TODO: This should be 'Fe' in the kepler progenitors. Double check that mass fractions add to 1.
        self.data['co56'] = np.zeros(self.cell_count) # Missing in progenitor data?
        self.data['ni56'] = np.zeros(self.cell_count) 



    def write_mesa_input(self, model_index):
        '''Writes the star's data into MESA input files.''' 
        
        # Easy way to convert values into the correct format for MESA model files
        def format_number(number):
            return f"{number:.16e}".replace('e', 'D')
        
        # IMPLEMENTED
        n_shells = ' ' * (25 - int(np.floor(np.log10(self.cell_count)))) + str(self.cell_count)
        total_mass = np.sum(self.mass_density * self.volume) / M_sun # In units of solar mass
        total_energy = np.sum(self.total_energy) 
        model_number = ' ' * (25 - int(np.floor(np.log10(model_index)))) + str(model_index)

        # TODO: UNIMPLEMENTED, possibly unnecessary
        star_age = 6.3376175628057906E-09
        initial_z = 2.0000000000000000E-02
        R_center = 3.9992663820663236E+07 # Look to flash_to_snec code
        Teff = 7.2257665281116360E+03 # Use the stefan-boltzmann law after luminosity is calculated
        xmstar = 2.0363416138072876E+34 # How do I calcualte this?
        core_mass = 3.0136524023699760E+33 # Same method as R_center

        file_header = f"""! note: initial lines of file can contain comments
    !
            548 -- model for mesa/star, cell center Riemann velocities (u), mlt convection velocity (mlt_vc). cgs units. lnd=ln(density), lnT=ln(temperature), lnR=ln(radius), L=luminosity, dq=fraction of xmstar=(mstar-mcenter) in cell; remaining cols are mass fractions.

                    version_number   'r24.08.1'
                            M/Msun      {format_number(total_mass)}
                    model_number      {model_number}
                        star_age      {format_number(star_age)}
                        initial_z      {format_number(initial_z)}
                        n_shells      {n_shells}
                        net_name   'approx21_cr60_plus_co56.net'
                            species                              22
                            xmstar      {format_number(xmstar)}  ! above core (g).  core mass: Msun, grams:      {format_number(core_mass / M_sun)}    {format_number(core_mass)}
                        R_center      {format_number(R_center)}  ! radius of core (cm).  R/Rsun, avg core density (g/cm^3):      {format_number(R_center / R_sun)}    1.1247695543683205D+10
                            Teff      {format_number(Teff)}
                    power_nuc_burn      0.0000000000000000D+00
                    power_h_burn      0.0000000000000000D+00
                    power_he_burn      0.0000000000000000D+00
                    power_z_burn      0.0000000000000000D+00
                        power_photo      0.0000000000000000D+00
                    total_energy      {format_number(total_energy)}
            cumulative_energy_error      1.3995525796409981D+42
    cumulative_error/total_energy      1.7494407245495471D-09  log_rel_run_E_err     -8.7571007679208037D+00
                        num_retries                               0

                lnd                        lnT                        lnR                          L                         dq                          u                     mlt_vc                   neut                       h1                         prot                       he3                        he4                        c12                        n14                        o16                        ne20                       mg24                       si28                       s32                        ar36                       ca40                       ti44                       cr48                       cr60                       fe52                       fe54                       fe56                       co56                       ni56     
    """

        # Data that will be written in table form in the stir_output.mod file
        # Also reverses the order of rows in the table so that the first cell is the outer radius and last cell is the center
        mesa_columns = ['lnd', 'lnT', 'lnR', 'L', 'dq', 'u', 'mlt_vc', 'neut', 'h1', 'prot', 'he3', 'he4', 'c12', 'n14', 'o16', 'ne20', 
                        'mg24', 'si28', 's32', 'ar36', 'ca40', 'ti44', 'cr48', 'cr60', 'fe52', 'fe54', 'fe56', 'co56', 'ni56', ]
        self.mesa_table_data = pd.DataFrame(self.data, columns = mesa_columns)
        self.mesa_table_data = self.mesa_table_data.iloc[::-1]  

        # Add one line for each cell, consisting of all it's properties
        new_lines = []
        for line_index in range(self.cell_count):

            # Writes the cell/line index 
            spaces = 4 - int(np.floor(np.log10(line_index + 1)))
            new_line = ' ' * spaces + str(line_index + 1)

            # Writes each of the properties
            for column_name in self.mesa_table_data.columns:
                spaces = 5 if self.mesa_table_data.at[line_index, column_name] > 0 else 4
                new_line += spaces * ' ' + format_number(self.mesa_table_data.at[line_index, column_name])

            new_lines.append(new_line)

        # Footer containing info about the previous model
        file_footer = f"""
        
        previous model

                previous n_shells      {n_shells}
            previous mass (grams)      2.3377068540442852D+34 
                timestep (seconds)      4.2684831294390047D-04 
                dt_next (seconds)      5.1221797553268049D-04 """

        # Write all of the above to the stir_output.mod file
        with open(f'model_{model_index}/stir_output.mod', 'w') as file:
            file.writelines(file_header + '\n'.join(new_lines) + file_footer)



def read_kepler_progenitor(progenitor_mass):
    '''
    Reads in kepler progenitor data, using the progs package, and returns a dataframe.
    Also ensures the returned data has the same column names as the STIR output for compatibility purposes.
    '''
    prog_data = ProgModel(str(progenitor_mass), "sukhbold_2016").profile
    prog_data.rename(columns={"radius_edge": "r", "temperature": "temp", "neutrons": "neut", "luminosity": "L", "energy": "ener"}, inplace=True)
    print(prog_data.columns.to_list())
    # TODO: May need to convert velocity to the cell center riemann velocity
    # TODO: Need to calcualte gravitational potential # Would this just be V = -GM/r where r is radius of the shell/cell?
    return prog_data



def read_mesa_output(path):
    '''
    Reads the output data from a MESA model and stores the necessary variables.
    Also ensures the returned data has the same column names as the STIR output for compatibility purposes.
    '''
    
    with open(path, "r") as file:

        # Find the start of the table data
        lines = file.readlines()
        for i, line in enumerate(lines):
            if "lnd" in line and "lnT" in line and "model for mesa/star" not in line:
                table_start = i
                break
        
        # Find the end of the table data
        for i, line in enumerate(lines[table_start + 1:]):
            if line == "\n":
                table_end = table_start + i + 1
                break
            
            # Makes each line of table data easier to parse
            lines[table_start + i + 1] = line.replace("    ", ",").replace(" ", "").replace("\n", "").replace("D", "e")
            if lines[table_start + i + 1][0] == ",": lines[table_start + i + 1] = lines[table_start + i + 1][1:]

        # Extract headers and numerical data

        # Process the numerical data into a pandas dataframe
        numerical_data = lines[table_start + 1:table_end]
        structured_data = [list(map(float, line.split(",")[1:])) for line in numerical_data]
        column_headers = ['density', 'temp', 'r', 'L', 'dq', 'u', 'mlt_vc', 'neut', 'h1', 'prot', 
                          'he3', 'he4', 'c12', 'n14', 'o16', 'ne20', 'mg24', 'si28', 's32', 'ar36', 
                          'ca40', 'ti44', 'cr48', 'cr60', 'fe52', 'fe54', 'fe56', 'co56', 'ni56']
        prog_data = pd.DataFrame(structured_data, columns=column_headers)

        # Convert the ln values for easier stitching with STIR output later
        prog_data["density"] = np.e ** prog_data['density']
        prog_data["temp"] = np.e ** prog_data['temp']
        prog_data["r"] = np.e ** prog_data['r']

        # TODO: Need to calculate total specific energy of each cell
        # TODO: Need to calculate gravitational potential of each cell
        # prog_data = prog_data.assign(ener = )
        # prog_data = prog_data.assign(gpot = ) # Would this just be V = -GM/r where r is radius of the shell/cell?

        # STIR uses ascending order with radius, while MESA uses descending order
        # So, this returns the MESA ouput but in ascending order to match STIR
        return prog_data.iloc[::-1]  



def read_stir_checkpoint(path):
    '''Reads in data from a STIR checkpoint file, returning it as a dataframe.'''
    stir_data = yt.load(path).all_data()
    grab_data = [("gas", "density"), ("flash", "temp"), ("gas", "r"), ("flash", "velx"), ("gas", "pressure"), ("flash", "ye  "),
                 ("flash", "cell_volume"), ("flash", "ener"), ("flash", "gpot"), ("gas", "gravitational_potential")]
    return stir_data.to_dataframe(grab_data)