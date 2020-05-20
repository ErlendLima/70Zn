from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt 
import shellmodelutilities as smutil



# Set bin width and range
bin_width = 0.20
Emax = 14
Nbins = int(np.ceil(Emax/bin_width))
Emax_adjusted = bin_width*Nbins # Trick to get an integer number of bins
bins = np.linspace(0,Emax_adjusted,Nbins+1)

# Define list of calculation input files and corresponding label names
inputfile = "summary_Zn70_jun45.txt"

# Instantiate figure which we will fill
f_rho, ax_rho = plt.subplots(1,1)

# Read energy levels from file
levels = smutil.read_energy_levels(inputfile)

# Choose which [2*J,pi] combinations to include in partial level density plot
Jpi_list = [[0,-1],[2,-1],[4,-1],[6,-1],[8,-1],[10,-1],[12,-1],[14,-1],[16,-1],[18,-1],[20,-1],[22,-1],[24,-1],[26,-1],[28,-1],
            [0,+1],[2,+1],[4,+1],[6,+1],[8,+1],[10,+1],[12,+1],[14,+1],[16,+1],[18,+1],[20,+1],[22,+1],[24,+1],[26,+1],[28,+1]]

# Allocate (Ex,Jpi) matrix to store partial level density
rho_ExJpi = np.zeros((Nbins,len(Jpi_list)))
# Count number of levels for each (Ex, J, pi) pixel.
Egs = levels[0,0] # Ground state energy
for i_l in range(len(levels[:,0])):
    E, J, pi = levels[i_l]
    # Skip if level is outside range:
    if E-Egs >= Emax:
        continue
    i_Ex = int(np.floor((E-Egs)/bin_width))
    try:
        i_Jpi = Jpi_list.index([J,pi])
    except:
        continue
    rho_ExJpi[i_Ex,i_Jpi] += 1

rho_ExJpi /= bin_width # Normalize to bin width, to get density in MeV^-1


# Plot it
from matplotlib.colors import LogNorm # To get log scaling on the z axis
colorbar_object = ax_rho.pcolormesh(np.linspace(0,len(Jpi_list)-1,len(Jpi_list)), bins, rho_ExJpi, norm=LogNorm())
f_rho.colorbar(colorbar_object) # Add colorbar to plot

# Make the plot nice
ax_rho.set_xlabel(r"$\pi\cdot J\,\mathrm{(\hbar)}$")
ax_rho.set_ylabel(r'$E_x \, \mathrm{(MeV)}$')

# A bit of Python voodoo to get the x labels right:
Jpi_array = np.append(np.linspace(0,-int((len(Jpi_list)-1)/2),int(len(Jpi_list)/2)),np.linspace(0,int((len(Jpi_list)-1)/2),int(len(Jpi_list)/2))) # Array of pi*J for plot
def format_func(value, tick_number):
    if value >= 0 and value <= 28:
        return int(Jpi_array[int(value)])
    else:
        return None
ax_rho.set_xlim([0,29])
ax_rho.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax_rho.set_xticks([0,2,4,6,8,10,12,14,15,17,19,21,23,25,27])

# Show plot
plt.show()
