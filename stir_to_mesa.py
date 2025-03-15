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

# Constants in CGS units
M_sun = 1.989E33 # Mass of the sun in grams
R_sun = 6.959E10 # Radius of the sun in centimeters
sigma_b = 5.669E-5 # Stefan-Boltzmann constant

nuclear_network = ['neut', 'h1', 'prot', 'he3', 'he4', 'c12', 'n14', 'o16', 'ne28', 'mg24', 'si28', 's32', 'ar36', 'ca48', 'ti44', 'cr48', 'cr60', 'fe52', 'fe54', 'fe56', 'co56', 'ni56']



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
        '''Reads data from a STIR checkpoint at the path and stores the necessary variables.'''

        # Finds the first index in the progenitor data that is outside the STIR domain
        prog_domain = len(prog_data.loc[prog_data['enclosed_mass'].values > np.max(stir_data['enclosed_mass'].values)])
        self.stitch_index = prog_data.shape[0] - prog_domain

        # TODO: Progenitors are missing gpot
        # Combines STIR and progenitor data, interpolating STIR values based by enclosed mass
        self.data = prog_data.copy()
        for col in stir_data.columns:
            if col == "enclosed_mass": continue
            
            # And If the column doesn't exist in progenitor data, notify us and skip it
            if not(col in self.data):
                print(f"Column {col} missing from progenitor data.")
                continue

            # If this data is composition, just overwrite with the progenitor values since STIR doesn't simulate the nuclear network
            # TODO: This is temporary, as composition should definitely change during the STIR simulation
            if col in nuclear_network:
                self.data.loc[0:self.stitch_index - 1, col] = prog_data['enclosed_mass'].values[:self.stitch_index]
                continue

            # Otherwise, regrid the STIR data to fit the progenitor mass coordinates
            interp_data = np.interp(prog_data['enclosed_mass'].values[:self.stitch_index], stir_data['enclosed_mass'].values, stir_data[col])
            self.data.loc[0:self.stitch_index - 1, col] = interp_data

        # Calculate the total specific energy, including gravitational potential
        #self.total_specific_energy = full_data['ener'].values - 1.275452534480232E+018 + full_data['gpot'].values

        # Stitches the progenitor data onto the STIR ouput
        self.plot_data(self.data['enclosed_mass'].values, ylog = True, xlabel = "Cell Index", ylabel = "log(enclosed mass)")
        self.plot_data(self.data['r'].values, ylog = True, xlabel = "Cell Index", ylabel = "log(radius)")
        self.plot_data(self.data['density'].values, ylog = True, xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass (M_sun)", ylabel = "log(mass density)")
        self.plot_data(self.data['pressure'].values, ylog = True, xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass (M_sun)", ylabel = "log(pressure)")
        self.plot_data(self.data['temp'].values, ylog = True, xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass (M_sun)", ylabel = "log(temp)")
        self.plot_data(self.data['cell_volume'], xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass (M_sun)", ylabel = "volume")
        #self.total_specific_energy = plot_data(self.total_specific_energy, prog_total_specific_energy, stitch_index, ylog = False, xaxis = self.enclosed_mass, xlabel = "log(mass)", ylabel = "total specific energy")

        # TODO: MESA says it uses edge velocity, but MESA model files need cell-center riemann velocities?
        self.plot_data(self.data['velx'], xaxis = self.data['enclosed_mass'].values, xlabel = "enclosed mass", ylabel = "radial velocity")

        # Calculating the total energy of each cell in the star
        #self.total_energy = (stir_data['ener'] - 1.275452534480232E+018 + stir_data['gpot']) * stir_data['density'].v  * stir_data['cell_volume']

        # DEBUG: Plot the total specific energy with respect to log(radius) to find what a "small amount of energy" might be
        #plt.plot(np.log10(self.radius[:len(stir_data['density'].values)]), self.total_specific_energy)
        #plt.axhline(0, ls = "--", c = "red")
        #plt.axhline(pns_boundary_energy, ls = "--", c = "green")
        #plt.ylim(-0.05e20, 0.15e20)

        # Estimate the PNS boundary as the point where the total specific energy is first above some small energy (such as 2e18)
        #pns_index = np.min(np.where(self.total_specific_energy < pns_boundary_energy))

        # Calculate the overburden energy as the total energy of all cells past the estimated PNS boundary
        # TODO: This needs to include the progenitor domain as well but we need to calcualte gpot to get total specific energy for the progenitors
        #overburden_energy = np.sum(self.total_specific_energy[pns_index:])

        # Go back to the initial PNS index and integrate the specific total energy with a changing mass until the sum is equal to the overburden energy.
        # The index at that point is where the PNS star should stop.
        
        # Get the total mass of the proto-neutron star
        #self.pns_masscut = np.sum(self.mass_density[:pns_index] * self.volume[:pns_index])
        
        # Structuring the data for each writing of MESA model files
        # TODO: With tweaks, we can skip the below table, instead adding the ln() values to 
        self.mesa_table = {}
        self.mesa_table['lnR'] = np.log(self.data['r'].values)
        self.mesa_table['lnd'] = np.log(self.data['density'].values)
        self.mesa_table['lnT'] = np.log(self.data['temp'].values)
        self.mesa_table['L'] = np.zeros(self.data.shape[0]) # Luminosity TODO: Should I calculate this using the total energy and some dynamical time?
        self.mesa_table['dq'] = np.zeros(self.data.shape[0]) # TODO: Fraction of xmstar=(mstar-mcenter) - what does this mean?
        self.mesa_table['u'] = np.zeros(self.data.shape[0]) # TODO: Cell center riemann velocity - Is this simply the velocity of each edge averaged?
        self.mesa_table['mlt_vc'] = np.zeros(self.data.shape[0]) # TODO: MLT convection velocity - How can I calculate this?
        self.mesa_table['neut'] = np.zeros(self.data.shape[0]) # neutrons in progenitor data
        self.mesa_table['h1'] = np.zeros(self.data.shape[0])
        self.mesa_table['prot'] = np.zeros(self.data.shape[0]) # not tracked separately from h1 in progenitor data
        self.mesa_table['he3'] = np.zeros(self.data.shape[0])
        self.mesa_table['he4'] = np.zeros(self.data.shape[0])
        self.mesa_table['c12'] = np.zeros(self.data.shape[0])
        self.mesa_table['n14'] = np.zeros(self.data.shape[0])
        self.mesa_table['o16'] = np.zeros(self.data.shape[0])
        self.mesa_table['ne20'] = np.zeros(self.data.shape[0]) 
        self.mesa_table['mg24'] = np.zeros(self.data.shape[0])
        self.mesa_table['si28'] = np.zeros(self.data.shape[0]) 
        self.mesa_table['s32'] = np.zeros(self.data.shape[0])
        self.mesa_table['ar36'] = np.zeros(self.data.shape[0]) 
        self.mesa_table['ca40'] = np.zeros(self.data.shape[0])
        self.mesa_table['ti44'] = np.zeros(self.data.shape[0])
        self.mesa_table['cr48'] = np.zeros(self.data.shape[0])
        self.mesa_table['cr60'] = np.zeros(self.data.shape[0]) # Missing in progenitor data?
        self.mesa_table['fe52'] = np.zeros(self.data.shape[0]) 
        self.mesa_table['fe54'] = np.zeros(self.data.shape[0])
        self.mesa_table['fe56'] = np.zeros(self.data.shape[0]) # TODO: This should be 'Fe' in the kepler progenitors. Double check that mass fractions add to 1.
        self.mesa_table['co56'] = np.zeros(self.data.shape[0]) # Missing in kepler progenitor data?
        self.mesa_table['ni56'] = np.zeros(self.data.shape[0]) 


    
    def plot_data(self, data, xaxis = None, xlabel = "", ylabel = "", xlog = False, ylog = False, zoom_width = 80):

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
        axs[0].axvline(used_xaxis[self.stitch_index], ls="--", c="red", alpha=0.3)

        # Plot again but zoomed in on the stitch region
        axs[1].plot(used_xaxis[self.stitch_index - zoom_width//2:self.stitch_index + zoom_width//2], plot_data[self.stitch_index - zoom_width//2:self.stitch_index + zoom_width//2])
        axs[1].axvline(used_xaxis[self.stitch_index], ls="--", c="red", alpha=0.3)
        axs[1].set_title(f"Stitch Region")
        
        plt.show()



    def write_mesa_input(self, model_index):
        '''Writes the star's data into MESA input files.''' 
        
        # Easy way to convert values into the correct format for MESA model files
        def format_number(number):
            return f"{number:.16e}".replace('e', 'D')
        
        # IMPLEMENTED
        n_shells = ' ' * (25 - int(np.floor(np.log10(self.data.shape[0])))) + str(self.data.shape[0])
        total_mass = np.sum(self.data['density'] * self.data['cell_volume']) / M_sun # In units of solar mass
        total_energy = 0#np.sum(self.total_energy) 
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
        mesa_input = pd.DataFrame(self.mesa_table, columns = mesa_columns).iloc[::-1]

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
        file_footer = f"""
        
        previous model

               previous n_shells      {n_shells}
           previous mass (grams)      2.3377068540442852D+34 
              timestep (seconds)      4.2684831294390047D-04 
               dt_next (seconds)      5.1221797553268049D-04"""

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

    # Calculates the progenitor volume
    prog_volume = (4/3) * np.pi * np.diff(np.concatenate(([0], prog_data['r'].values**3)))
    prog_data = prog_data.assign(cell_volume = prog_volume) 
    prog_data = prog_data.assign(enclosed_mass = np.cumsum(prog_data['cell_volume'].values * prog_data['density'].values) / M_sun)
    #print(prog_data.columns.to_list())
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

        # Calculates the progenitor volume
        prog_volume = (4/3) * np.pi * np.diff(np.concatenate(([0], prog_data['r'].values**3)))
        prog_data = prog_data.assign(cell_volume = prog_volume) 
        prog_data = prog_data.assign(enclosed_mass = np.cumsum(prog_data['cell_volume'].values * prog_data['density'].values) / M_sun)

        # TODO: Need to calculate total specific energy of each cell
        # TODO: Need to calculate gravitational potential of each cell
        # prog_data = prog_data.assign(ener = )
        # prog_data = prog_data.assign(gpot = ) # Would this just be V = -GM/r where r is radius of the shell/cell?

        # STIR uses ascending order with radius, while MESA uses descending order
        # So, this returns the MESA ouput but in ascending order to match STIR
        return prog_data.iloc[::-1]  



def read_stir_checkpoint(path):
    '''Reads in data from a STIR checkpoint file, making modifications and returning it as a dataframe.'''
    stir_data = yt.load(path).all_data()
    grab_data = [("gas", "density"), ("flash", "temp"), ("gas", "r"), ("flash", "velx"), ("gas", "pressure"), ("flash", "ye  "),
                 ("flash", "cell_volume"), ("flash", "ener"), ("flash", "gpot"), ("gas", "gravitational_potential")]
    stir_dataframe = stir_data.to_dataframe(grab_data)
    stir_dataframe['r'] = shift_to_cell_edge(stir_dataframe['r'].values) # Shifts radius to the cell edge to match progenitor outputs
    stir_dataframe['velx'] = shift_to_cell_edge(stir_dataframe['velx'].values) # Shifts radial velocity to the cell edge to match progenitor outputs
    stir_dataframe = stir_dataframe.assign(enclosed_mass = np.cumsum(stir_dataframe['cell_volume'].values * stir_dataframe['density'].values) / M_sun)

    # Add nuclear network with all zeros for mass fractions
    for col in nuclear_network:
        stir_dataframe[col] = stir_dataframe.shape[0]

    return stir_dataframe