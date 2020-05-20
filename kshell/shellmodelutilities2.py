
import numpy as np
import re
# import matplotlib.pyplot as plt
# import sys

# Scripts to sort levels and transition strengths from KSHELL
# into Ex-Eg matrix energy bins, and to make level density and
# gamma strength function.
# Plus other useful functions for shell model stuff.


def div0(a, b):
    """ division function designed to ignore / 0, i.e. div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0  # -inf inf NaN
    return c


def read_energy_levels(inputfile):
    # Reads levels from a KSHELL summary file, returns Nx3 matrix of
    # [Ei, 2*Ji, parity], where E is absolute energy of level and parity 1=+, 0=-
    levels = []
    with open(inputfile, 'r') as f:
        lines = f.readlines()
        i_start = -1
        for i in range(len(lines)):
            if len(lines[i].split()) < 1:
                continue
            if lines[i].split()[0] == "Energy":
                i_start = i+4
                break
        for i in range(i_start, len(lines)):
            if len(lines[i].split()) < 1:
                break
            words = lines[i].split()
            num_parity = 1 if words[2] == "+" else -1
            levels.append([float(words[5]), float(words[1]), num_parity])
        return np.asarray(levels)


def read_transition_strengths(inputfile, type="M1"):
    transitions = []
    with open(inputfile, 'r') as f:
        lines = f.readlines()
        i_start = -1
        for i in range(len(lines)):
            if len(lines[i].split()) < 1:
                continue
            if lines[i].split()[0] == "B({:s})".format(type):
                # print "hello"
                i_start = i+3
                break
        for i in range(i_start, len(lines)):
            # print lines[i]
            if len(lines[i].split()) < 1:
                break
            line = lines[i]
            # Returns
            # [2Ji, pi, Ei, 2Jf, pf, Ef, Eg, B(M1,i->f)]  --  beware that the summary file has opposite initial/final convention to this!
            # JEM: Changed 20170428 from [2Ji, Ei, 2Jf, Ef, Eg, B(M1,i->f)]
            # print line[0:3], line[12:22], line[22:25], line[34:43], line[43:51], line[51:67]
            # print line[25]
            print(line)
            pi_str = line[25:27].strip()
            # print pi_str
            if pi_str == "+":
                pi = +1
            elif pi_str == "-":
                pi = -1
            else:
                raise Exception(
                    "From function read_transition_strengths: Could not assign initial parity. Read value on line: "+pi_str+" "+str(i))
            pf_str = line[4:5].strip()
            if pf_str == "+":
                pf = +1
            elif pf_str == "-":
                pf = -1
            else:
                raise Exception(
                    "From function read_transition_strengths: Could not assign final parity Read value on line: "+pf_str+" "+str(i))
            transitions.append([float(line[22:25]), pi, float(line[34:43]), float(
                line[0:3]), pi, float(line[12:22]), float(line[43:51]), float(line[67:83])])
        return np.asarray(transitions)


def read_transition_strengths_2(inputfile, type="M1"):
    transitions = []
    stream = open(inputfile)
    line = stream.readline()
    while line:
        line = stream.readline()
        if not line:
            raise RuntimeError("Could not find B(M1)")

        if line.startswith("B({})".format(type)):
            break

    # Parse the B(M1). The file format depends on the KSHELL version
    # Handle at least two versions
    line = stream.readline()
    if "mu_N^2" in line:
        transitions = parse_transitions(stream)
    elif "2Ji" in line:
        transitions = parse_transitions_old(stream)
    else:
        raise RuntimeError("Don't know how to parse transitions.")
    stream.close()
    return transitions


def parse_transitions_old(stream):
    transitions = []
    line = stream.readline()
    while len(line) > 1:
        # Split each line at white space not contained within a ( )
        tokens = re.split(r'\s+(?![^()]*\))', line.rstrip().strip())
        Ji = float(tokens[0])/2   # Initial spin and parity
        Jf = float(tokens[4])/2   # Final --- || ---
        pi_i = 1 if tokens[1] == '+' else -1
        pi_f = 1 if tokens[5] == '+' else -1
        assert pi_i == pi_f, "Initial and final parity do not match"
        Ei = tokens[3]
        Ef = tokens[7]
        dE = tokens[8]
        BM1if = tokens[8]
        transition = [Ji, pi_i, Ei, Jf, pi_f, Ef, dE, BM1if]
        transitions.append([float(x) for x in transition])

        line = stream.readline()

    return np.asarray(transitions)


def parse_transitions(stream):
    transitions = []

    stream.readline()
    line = stream.readline()
    while len(line) > 1:
        # Split each line at white space not contained within a ( )
        line = line.rstrip().strip()
        tokens = re.split(r'\s+(?![^()]*\))', line)
        init = tokens[0]   # Initial spin and parity
        final = tokens[2]  # Final --- || ---
        Ji, pi_i = re.split(r'[+-]', init)
        Jf, pi_f = re.split(r'[+-]', final)
        pi_i = 1 if pi_i == '+' else -1
        pi_f = 1 if pi_f == '+' else -1
        assert pi_i == pi_f, "Initial and final parity do not match"
        Ei = tokens[1]
        Ef = tokens[3]
        dE = tokens[4]
        BM1if = tokens[5].split('(')[0]
        transition = [Ji, pi_i, Ei, Jf, pi_f, Ef, dE, BM1if]
        transitions.append([float(x) for x in transition])

        line = stream.readline()

    return np.asarray(transitions)


def total_level_density(levels, bin_width, Ex_max):
    # 20170816: This function returns the total level density as a function of Ex.
    # Make sure the number of bins cover the whole Ex region.
    Nbins = int(np.ceil(Ex_max/bin_width))
    # Array of lower bin edge energy values
    bin_array = np.linspace(0, bin_width*Nbins, Nbins+1)
    Egs = levels[0, 0]
    rho_total, tmp = np.histogram(levels[:, 0]-Egs, bins=bin_array)
    rho_total = rho_total/bin_width  # To get density per MeV
    return rho_total


def strength_function_average2(levels, transitions, Jpi_list, bin_width, Ex_min, Ex_max, type="M1"):
    """
    20171009: Updated the way we average over Ex, J, pi
              to only count pixels with non-zero gSF.
     20170815: This function returns the strength function
               by taking the partial level density
               corresponding to the specific (Ex, J, pi) pixel in the
    calculation of the strength function,
    and then averaging over all three variables to produce
     <f(Eg)>.
     Make sure the number of bins cover the whole Ex region.
    """
    Nbins = int(np.ceil(Ex_max/bin_width))
    print(Nbins)

    # Array of lower bin edge energy values
    bin_array = np.linspace(0, bin_width*Nbins, Nbins+1)
    # Array of middle bin values
    bin_array_middle = (bin_array[:-1]+bin_array[1:])/2

    # Find index of first and last bin (lower bin edge) where we put counts.
    # It's important to not include the other Ex bins in the averaging later,
    # because they contain zeros which will pull the average down.
    i_Exmin = int(np.floor(Ex_min/bin_width))
    i_Exmax = int(np.floor(Ex_max/bin_width))

    prefactor = {"M1":  11.5473e-9, "E1": 1.047e-6}

    # Read out the absolute ground state energy,
    # so we can get relative energies later
    Egs = levels[0, 0]

    # Allocate matrices to store the summed B(M1) values for each pixel,
    # and the number of transitions counted
    B_pixel_sum = np.zeros((Nbins, Nbins, len(Jpi_list)))
    B_pixel_count = np.zeros((Nbins, Nbins, len(Jpi_list)))

    # Loop over all transitions and put in the correct pixel:
    for i_tr in range(len(transitions[:, 0])):
        Ex = transitions[i_tr, 2]  - Egs
        # Check if transition is below Ex_max, skip if not
        if not (Ex_min < Ex <= Ex_max):
            continue

        # Get bin index for Eg and Ex (initial).
        # Indices are defined with respect to the lower bin edge.
        i_Eg = int(np.floor(transitions[i_tr, 6]/bin_width))
        i_Ex = int(np.floor(Ex/bin_width))

        # Read initial spin and parity of level:
        Ji = int(transitions[i_tr, 0])
        pi = int(transitions[i_tr, 1])
        try:
            i_Jpi = Jpi_list.index([Ji, pi])
        except Exception:
            print(f"Could not find [{Ji}, {pi}] in spin list")
            continue
        print("Found", i_Jpi)

        # Add B(M1) value and increment count to pixel, respectively
        B_pixel_sum[i_Ex, i_Eg, i_Jpi] += transitions[i_tr, 7]

        B_pixel_count[i_Ex, i_Eg, i_Jpi] += 1  # Original

    # Allocate (Ex,Jpi) matrix to store level density
    rho_ExJpi = np.zeros((Nbins, len(Jpi_list)))
    # Count number of levels for each (Ex, J, pi) pixel.
    for i_l in range(len(levels[:, 0])):
        E, J, pi = levels[i_l]
        # Skip if level is outside range:
        if E-Egs >= Ex_max:
            continue
        i_Ex = int(np.floor((E-Egs)/bin_width))
        try:
            i_Jpi = Jpi_list.index([J, pi])
        except Exception:
            continue
        rho_ExJpi[i_Ex, i_Jpi] += 1

    rho_ExJpi /= bin_width  # Normalize to bin width, to get density in MeV^-1

    # Calculate gamma strength functions for each Ex, J, pi individually, using the partial level density for each J, pi.
    gSF = np.zeros((Nbins, Nbins, len(Jpi_list)))
    a = prefactor[type]  # mu_N^-2 MeV^-2, conversion constant
    for i_Jpi in range(len(Jpi_list)):
        for i_Ex in range(Nbins):
            gSF[i_Ex, :, i_Jpi] = a * div0(B_pixel_sum[i_Ex, :, i_Jpi],
                                           B_pixel_count[i_Ex, :, i_Jpi]) * rho_ExJpi[i_Ex, i_Jpi]
    gSF_currentExrange = gSF[i_Exmin:i_Exmax+1, :, :]
    gSF_ExJpiavg = div0(gSF_currentExrange.sum(axis=(0, 2)),
                        (gSF_currentExrange != 0).sum(axis=(0, 2)))
    return gSF_ExJpiavg


def strength_function_average(levels, transitions, Jpi_list, bin_width, Ex_min, Ex_max, type="M1"):
    # 20171009: Updated the way we average over Ex, J, pi to only count pixels with non-zero gSF.
    # 20170815: This function returns the strength function by taking the partial level density
    # corresponding to the specific (Ex, J, pi) pixel in the
    # calculation of the strength function, and then averaging over all three variables to produce
    # <f(Eg)>.
    # Make sure the number of bins cover the whole Ex region.
    Nbins = int(np.ceil(Ex_max/bin_width))
    # print "Ex_max =", Ex_max, "Nbins*bin_width =", Nbins*bin_width
    # Array of lower bin edge energy values
    bin_array = np.linspace(0, bin_width*Nbins, Nbins+1)
    # Array of middle bin values
    bin_array_middle = (bin_array[:-1]+bin_array[1:])/2
    # print(bin_array_middle)
    # Find index of first and last bin (lower bin edge) where we put counts.
    # It's important to not include the other Ex bins in the averaging later, because they contain zeros which will pull the average down.
    i_Exmin = int(np.floor(Ex_min/bin_width))
    i_Exmax = int(np.floor(Ex_max/bin_width))

    prefactor = {"M1":  11.5473e-9, "E1": 1.047e-6}

    # Read out the absolute ground state energy, so we can get relative energies later
    Egs = levels[0, 0]

    # Allocate matrices to store the summed B(M1) values for each pixel, and the number of transitions counted
    B_pixel_sum = np.zeros((Nbins, Nbins, len(Jpi_list)))
    B_pixel_count = np.zeros((Nbins, Nbins, len(Jpi_list)))

    # Loop over all transitions and put in the correct pixel:
    for i_tr in range(len(transitions[:, 0])):
        Ex = transitions[i_tr, 2] - Egs
        # Check if transition is below Ex_max, skip if not
        if Ex < Ex_min or Ex >= Ex_max:
            continue

        # Get bin index for Eg and Ex (initial). Indices are defined with respect to the lower bin edge.
        i_Eg = int(np.floor(transitions[i_tr, 6]/bin_width))
        i_Ex = int(np.floor(Ex/bin_width))

        # Read initial spin and parity of level:
        Ji = int(transitions[i_tr, 0])
        pi = int(transitions[i_tr, 1])
        try:
            i_Jpi = Jpi_list.index([Ji, pi])
        except:
            continue

        # Add B(M1) value and increment count to pixel, respectively
        B_pixel_sum[i_Ex, i_Eg, i_Jpi] += transitions[i_tr, 7]

        B_pixel_count[i_Ex, i_Eg, i_Jpi] += 1  # Original

    # Allocate (Ex,Jpi) matrix to store level density
    rho_ExJpi = np.zeros((Nbins, len(Jpi_list)))
    # Count number of levels for each (Ex, J, pi) pixel.

    for i_l in range(len(levels[:, 0])):
        E, J, pi = levels[i_l]
        # Skip if level is outside range:
        if E-Egs >= Ex_max:
            continue
        i_Ex = int(np.floor((E-Egs)/bin_width))
        try:
            i_Jpi = Jpi_list.index([J, pi])
        except:
            continue
        rho_ExJpi[i_Ex, i_Jpi] += 1

    rho_ExJpi /= bin_width  # Normalize to bin width, to get density in MeV^-1

    # Calculate gamma strength functions for each Ex, J, pi individually, using the partial level density for each J, pi.
    gSF = np.zeros((Nbins, Nbins, len(Jpi_list)))
    a = prefactor[type]  # mu_N^-2 MeV^-2, conversion constant
    for i_Jpi in range(len(Jpi_list)):
        for i_Ex in range(Nbins):
            gSF[i_Ex, :, i_Jpi] = a * div0(B_pixel_sum[i_Ex, :, i_Jpi],
                                           B_pixel_count[i_Ex, :, i_Jpi]) * rho_ExJpi[i_Ex, i_Jpi]
    gSF_currentExrange = gSF[i_Exmin:i_Exmax+1, :, :]
    gSF_ExJpiavg = div0(gSF_currentExrange.sum(axis=(0, 2)),
                        (gSF_currentExrange != 0).sum(axis=(0, 2)))
    return gSF_ExJpiavg


def level_density_matrix(inputfile, bin_width=0.2, Emax=12, Ex_low=5, Ex_high=8):

    levels = read_energy_levels(inputfile)

    # Set bin width and range
    Nbins = int(Emax/bin_width)
    Emax_adjusted = bin_width*Nbins
    bins_Ex = np.linspace(0, Emax_adjusted, Nbins+1)
    bins_Ex_middle = (bins_Ex[0:-1]+bins_Ex[1:])/2

    bins_J = np.linspace(0, levels[:, 1].max()/2, int(levels[:, 1].max()/2)+1)

    Egs = levels[0, 0]
    rho_total, tmp = np.histogram(levels[:, 0]-Egs, bins=bins_Ex)
    rho_total = rho_total/bin_width  # To get density per MeV

    matrix, xedges, yedges = np.histogram2d(
        levels[:, 1]/2, levels[:, 0]-Egs, bins=[bins_J, bins_Ex])
    return matrix, xedges, yedges


def level_density_matrix_parity_decomposed(inputfile, bin_width=0.2, Emax=12, Ex_low=5, Ex_high=8):

    levels = read_energy_levels(inputfile)

    # Set bin width and range
    Nbins = int(Emax/bin_width)
    Emax_adjusted = bin_width*Nbins
    bins_Ex = np.linspace(0, Emax_adjusted, Nbins+1)
    bins_Ex_middle = (bins_Ex[0:-1]+bins_Ex[1:])/2

    bins_J = np.linspace(0, levels[:, 1].max()/2, int(levels[:, 1].max()/2)+1)

    Egs = levels[0, 0]
    # rho_total, tmp = np.histogram(levels[:,0]-Egs, bins=bins_Ex)
    # rho_total = rho_total/bin_width # To get density per MeV

    matrix_plus, xedges, yedges = np.histogram2d(
        levels[levels[:, 2] > 0, 1]/2, levels[levels[:, 2] > 0, 0]-Egs, bins=[bins_J, bins_Ex])
    matrix_minus, xedges, yedges = np.histogram2d(
        levels[levels[:, 2] < 0, 1]/2, levels[levels[:, 2] < 0, 0]-Egs, bins=[bins_J, bins_Ex])
    return matrix_plus, matrix_minus, xedges, yedges


def spider(inputfiles, names, type="M1", threshold=0.1, spinwindow=[], Eg_low=0, Eg_high=1e9, Ex_low=0, Ex_high=1e9, scale=2):
    """
    Makes a "spider plot", a web of transitions between levels plotted as function of Ex and J.
    """

    Nsp = np.ceil(np.sqrt(len(inputfiles))).astype(int)
    f, ax_list = plt.subplots(Nsp, Nsp, squeeze=False,
                              sharex='col', sharey='row')

    for i in range(len(inputfiles)):
        inputfile = inputfiles[i]
        name = names[i]
        ax = ax_list[i % Nsp][int((i-i % Nsp)/Nsp)]

        levels = read_energy_levels(inputfile)
        Egs = levels[0, 0]

        Ex_high = min(Ex_high, levels[:, 0].max()-Egs)
        Eg_high = min(Eg_high, levels[:, 0].max()-Egs)

        levels_plot_J = []
        levels_plot_Ex = []
        for iEx in range(len(levels[:, 0])):
            # print levels[iEx,:]
            J2 = levels[iEx, 1]
            par = levels[iEx, 2]
            if len(spinwindow) > 0 and not ([J2, par] in spinwindow or [J2-2, par] in spinwindow or [J2+2, par] in spinwindow):
                continue
            Ex = levels[iEx, 0]-Egs
            if Ex < Ex_low or Ex > Ex_high:
                continue

            levels_plot_J.append(J2/2)
            levels_plot_Ex.append(Ex)

        ax.plot(levels_plot_J, levels_plot_Ex,
                'o', color='grey', linewidth=0.5)
        ax.set_xlim([levels[:, 1].min()/2-1, levels[:, 1].max()/2+1])
        ax.set_title(
            name+r'$\,E_\gamma\in[{:.1f},{:.1f}]$'.format(Eg_low, Eg_high))
        ax.set_ylabel(r'$E_x\,\mathrm{[MeV]}$')
        ax.set_xlabel(r'$J\,\mathrm{[\hbar]}$')

        transitions = read_transition_strengths(inputfile, type=type)
        for iEx in range(len(transitions[:, 0])):
            J2i = int(transitions[iEx, 0])
            # pari = int(transitions[
