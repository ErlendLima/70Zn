#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import datareader
import argparse
from matplotlib.colors import LogNorm, Normalize
from typing import Union, Any, Optional
from glob import glob
import sys

array = np.ndarray

class RuntimeFailure(Exception):
    def __init__(self, msg):
        self.args = [f"RuntimeError: {msg}"]
        sys.exit(self)


def div0(a, b):
    """ Ignore x/0, i.e. div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
    c[~ np.isfinite(c)] = 0
    return c


class Analyzer:
    def __init__(self, summary_file, *,
                 Ex_max: float = None,
                 Ex_min: float = 0.0,
                 bin_width: float = 0.2):
        self.summary_file = summary_file
        self.levels = datareader.read_energy_levels(summary_file)
        self.transitions = datareader.read_transition_strength(summary_file)
        self.bin_width = bin_width
        self.Ex_min = Ex_min
        self.Ex_max = self.levels['Ex'].max() if Ex_max is None else Ex_max

        Nbins = int(self.Ex_max/bin_width)
        self.Ex_max = bin_width*Nbins
        Ex = np.arange(Ex_min, self.Ex_max, bin_width)
        # Use middle bin
        Ex += bin_width/2

        print(f"{Nbins=}")
        print(f"{self.Ex_max=}")
        self.Ex = Ex

    @property
    def JiPi(self) -> [[int, int]]:
        Jpi = list(set(self.transitions['JiPi'].tolist()))
        return sorted(Jpi, key=lambda x: x[0])

    def partial_nld(self):
        J = np.sort(self.levels['J'].unique())
        J = np.append(J, J[-1]+1)
        # Left edge bins
        Ex = self.Ex - self.bin_width/2
        Ex = np.append(Ex, Ex[-1] + self.bin_width)

        # Something odd happens here. Don't know what
        # pos+neg != total, but total+shift
        #pi_pos = self.levels.query('Parity > 0')[['J', 'Ex']]
        #pi_neg = self.levels.query('Parity < 0')[['J', 'Ex']]
        #pos = np.histogram2d(pi_pos['J'], pi_neg['Ex'], bins=[J, Ex])[0] / self.bin_width
        #neg = np.histogram2d(pi_pos['J'], pi_neg['Ex'], bins=[J, Ex])[0] / self.bin_width
        Js = self.levels['J'].to_numpy()
        levels = self.levels['Ex'].to_numpy()
        nld, *_ = np.histogram2d(Js, levels, bins=[J, Ex])
        return (J, Ex), nld/self.bin_width

    def plot_partial_nld(self, ax=None, sum_pi=True, **kwargs):
        (J, Ex), nld = self.partial_nld()
        if ax is None:
            fig, ax = plt.subplots(nrows=1 if sum_pi else 2,
                                   sharex=True, sharey=True)

        if sum_pi:
            mesh = ax.pcolormesh(Ex[:-1], J[:-1], nld, norm=LogNorm(), **kwargs)
            ax.set_ylabel(f"$J [\hbar]$")
            ax.set_yticks(J-0.5)
            ax.set_yticklabels(J[:-1] - 1)
            ax.set_ylim(0, J[-2])
            #ax.figure.colorbar(mesh, ax=ax)
        else:
            ax[0].pcolormesh(Ex, J, pos, norm=LogNorm(), **kwargs)
            ax[1].pcolormesh(Ex, J, neg, norm=LogNorm(), **kwargs)

        return ax, mesh

    def total_nld(self):
        (_, Ex), nld = self.partial_nld()
        return Ex, nld.sum(axis=0)

    def plot_total_nld(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        Ex, nld = self.total_nld()

        ax.step(Ex[:-1], nld, where='mid')
        ax.set_yscale('log')
        ax.set_ylabel(r'$\rho(E_x)$ [MeV$^{-1}$]')
        return ax

    def plot_nld(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(nrows=2, sharex=True)
        else:
            fig = ax[0].figure
        _, mesh = self.plot_partial_nld(ax[0], **kwargs)
        self.plot_total_nld(ax[1])
        ax[1].set_xlabel(r'$E_x$ [MeV]')
        fig.subplots_adjust(left=0.11, right=0.85, hspace=0.1)
        cbax = fig.add_axes([0.86, 0.52, 0.02, 0.36])
        fig.colorbar(mesh, cax=cbax)
        ax[1].set_xlim(self.Ex[0], self.Ex[-6])
        return ax

    def old_total_level_density(self, bins: Optional[Union[float, int, array]] = None,
                            Ex_max: Optional[float] = None,
                            Ex_min: Optional[float] = None):
        """ Calculates total level density as a function of Ex """
        Ex_max = self.Ex_max if Ex_max is None else Ex_max
        Ex_min = self.Ex_min if Ex_min is None else Ex_min
        bin_width = self.bin_width
        Nbins = int(np.ceil(Ex_max/bin_width))
        if isinstance(bins, (float)):
            # Assume bin width
            Nbins = int(np.ceil(Ex_max/bins))
        elif isinstance(bins, (int)):
            # Assume nbins
            bin_width = (Ex_max - Ex_min)/Nbins
        elif isinstance(bins, (array)):
            bins = bins
        elif bins is not None:
            raise ValueError("Unsupported value for `bins`")

        if not isinstance(bins, (array)):
            bins = np.linspace(Ex_min, bin_width*Nbins, Nbins+1)
        bin_width = bins[1] - bins[0]
        print(f"{bin_width=}")

        # Array of lower bin edge energy values
        # [1, 2, 3, 4] => [1, 2) [2, 3) [3, 4]

        rho, tmp = np.histogram(self.levels['Ex'], bins=bins)
        rho = rho/bin_width  # To get density per MeV
        return bins, np.append(0, rho)

    def old_plot_total_level_density(self, **kwargs):
        bins, rho = self.total_level_density(**kwargs)
        fig, ax = plt.subplots()
        ax.step(bins, np.append(0, rho), where='pre')

        # Make the plot nice
        ax.set_yscale('log')
        ax.set_xlabel(r'$E_x \, \mathrm{(MeV)}$')
        ax.set_ylabel(r'$\rho \, \mathrm{(MeV^{-1})}$')

        # Show plot
        return (bins, rho), ax

    def gamma_strength_function(self, *, Jpi: [(float, float)] = None,
                                J: [float] = None, pi: (int) = (-1, 1),
                                bin_width: float = None,
                                Ex_min: float = None, Ex_max: float = None,
                                transition_type: str = 'M1'):
        prefactor = {"M1":  11.5473e-9, "E1": 1.047e-6}
        bin_width = self.bin_width if bin_width is None else bin_width
        Ex_min = self.Ex_min if Ex_min is None else Ex_min
        Ex_max = self.Ex_max if Ex_max is None else Ex_max
        if transition_type not in prefactor.keys():
            raise KeyError(f"Unsupported transition type: {transition_type}")
        if transition_type != 'M1':
            raise NotImplementedError()
        if Jpi is not None and J is not None:
            raise RuntimeError("Only a double Jpi list or a single Jpi list can be provided")
        if Jpi is None and J is None:
            Jpi = self.JiPi
            if 1 not in pi:
                Jpi = list(filter(lambda x: x[1] == -1, Jpi))
            if -1 not in pi:
                Jpi = list(filter(lambda x: x[1] == 1, Jpi))

        # Weave spin and parity together
        if Jpi is None:
            Jpi = list(product(J, pi))

        # TODO: BUG
        # Spin 0 messes things up. Don't know why
        # It is the Nbins. Maybe it should be Nbins+1 everywhere?
        Jpi = list(filter(lambda x: x[0] != 0, Jpi))

        Nbins = int(np.ceil(Ex_max/bin_width))
        bins = np.linspace(0, bin_width*Nbins, Nbins+1)

        # Filter the transitions for correct energy
        transitions = self.transitions.query("@Ex_min <= Ei and Ei < @Ex_max "
                                             "and JiPi in @Jpi")
        assert len(transitions) > 0, ("No transitions meet the conditions:\n"
                                      f"{Ex_min} < Ex < {Ex_max}\n"
                                      f"Jπ ∈ {Jpi}")

        # Find the index of the [spin, parity] in the given list
        i_Jpi = np.array([Jpi.index(j) for j in zip(transitions.Ji,
                                                    transitions.pi_i)])
        BM1 = transitions['BM1'].values

        # Find the bin index for each element
        Ex_digit = np.digitize(transitions['Ei'], bins, right=False)
        Eg_digit = np.digitize(transitions['dE'], bins, right=False)

        # Allocate matrices to store the summed B(M1) values for each pixel,
        # and the number of transitions counted
        B_pixel_sum = np.zeros((Nbins+1, Nbins+1, len(Jpi)))
        B_pixel_count = np.zeros((Nbins+1, Nbins+1, len(Jpi)))

        # Add the B(M1) values to the correct pixel and count
        np.add.at(B_pixel_sum, [Ex_digit, Eg_digit, i_Jpi], BM1)
        np.add.at(B_pixel_count, [Ex_digit, Eg_digit, i_Jpi],
                  np.ones(len(BM1)))

        B_pixel_avg = div0(B_pixel_sum, B_pixel_count)

        # Extract the partial level densities
        ρ = np.zeros((Nbins+1, len(Jpi)))

        levels = self.levels.query("Ex < @Ex_max and JPi in @Jpi")

        i_Jpi = [Jpi.index(j) for j in zip(levels.J, levels.Parity)]
        Ex_digit = np.digitize(levels['Ex'], bins, right=False)
        np.add.at(ρ, [Ex_digit, i_Jpi], np.ones(len(Ex_digit)))

        # Normalize to bin width, [MeV⁻¹]
        ρ /= bin_width

        # Calculate γSF for each (Ex, J, π) individually, using partial
        # level density for each (J, π)
        gSF = np.zeros((Nbins+1, Nbins+1, len(Jpi)))
        a = prefactor[transition_type]
        gSF = a * B_pixel_avg * ρ[:, np.newaxis, :]

        # Return average γSF(Eγ) over all (Ex, J, π)
        gSF_ExJpiavg = div0(gSF.sum(axis=(0, 2)),
                            (gSF != 0).sum(axis=(0, 2)))
        return bins, gSF_ExJpiavg

    def plot_gamma_strength_function(self, **kwargs):
        bins, gSF = self.gamma_strength_function(**kwargs)
        bins_middle = (bins[:-1]+bins[1:])/2
        fig, ax = plt.subplots()
        ax.semilogy(bins, gSF)
        ax.set_ylabel(r'$f\, \mathrm{(MeV^{-3})}$')
        ax.set_xlabel(r'$E_\gamma\,\mathrm{(MeV)}$')
        # ax.set_ylim(1e-12, 1e-8)
        ax.legend()
        plt.show()

    def plot_scheme(self):
        fig, ax = plt.subplots()
        prev = 0
        for i, (J, Pi, E, Ex, _) in self.levels.iterrows():
            if i < 10:
                ax.plot((0, 1), (Ex, Ex), 'k', alpha=0.1)
                pi = '+' if Pi > 0 else '-'
                ax.annotate(fr'${J:g}^{{{pi}}}$', xy=(1.2, 0.99*Ex))
                if i > 1 and abs(Ex-prev) < 0.1:
                    offset += 0.1
                else:
                    offset = 0
                ax.annotate(fr'${Ex*1e3:.0f}$ [keV]', xy=(1.3, 0.99*Ex+offset))
                prev = Ex
        ax.set_xlim(0, 2)
        return ax

    def plot_level_density(self):
        """ Play with changing bin_width and Ex_max """
        from matplotlib.widgets import Slider
        fig, ax = plt.subplots(ncols=2, sharey=True)
        fig.subplots_adjust(bottom=0.25, wspace=0)

        # Make the sliders
        axexmax = plt.axes([0.25, 0.1, 0.65, 0.03])
        axbinw = plt.axes([0.25, 0.15, 0.65, 0.03])

        Ex_max_slider = Slider(axexmax, r'$E_{x, \max}$',
                               0.1, self.levels['Ex'].max(),
                               valinit=20)
        binwidth_slider = Slider(axbinw, 'bin width',
                                 0.05, 1, valinit=0.2, valstep=0.01)

        # Make initial histogram
        bins, rho = self.total_level_density(bin_width=0.2, Ex_max=10)
        rho = np.append(0, rho)
        l, = ax[1].step(rho, bins)

        # Update histogram on change
        def update(val):
            Ex_max = Ex_max_slider.val
            bin_width = binwidth_slider.val
            bins, rho = self.total_level_density(bin_width, Ex_max)
            rho = np.append(0, rho)
            l.set_ydata(bins)
            l.set_xdata(rho)
            fig.canvas.draw_idle()
        binwidth_slider.on_changed(update)
        Ex_max_slider.on_changed(update)

        # Plot the levels
        for i, (J, Pi, E, Ex, _) in self.levels.iterrows():
            ax[0].plot((0, 1), (Ex, Ex), 'k', alpha=0.1)
        ax[0].set_ylim(0, 15)
        ax[1].set_xscale("log")

        plt.show()


def parse_arguments():
    desc = ("Performs some rudimentary analysis of KSHELL summary files")
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("summary_file",
                        help=("Name of the KSHELL summary file. Defaults "
                              "to searching the current directory, failing "
                              "if it finds fewer or more than one file."),
                        nargs='?', default=glob("summary_*.txt"))
    parser.add_argument('-j', nargs='+', type=int,
                        help="""The angular moment of the levels to use.
                              Defaults to a reasonable number of the
                              lowest available angular momenta""")
    parser.add_argument('-p', '--pi', nargs='+',
                        help=("The parity of the levels to use. Defaults "
                              "to both +/- if available."))
    parser.add_argument('--jpi', nargs='+',
                        help="""Specifies both the angular momentum and
                        parity of each level using a nested list on the
                        form [[J₁, π₁], [J₂, π₂], ...]""")
    parser.add_argument('-a', '--actions', nargs='+',
                        help="Specifies which analysis to run",
                        type=str.lower,
                        choices=['scheme', 'ld', 'g', 'gamma', 'ldplay'],
                        default=['ld', 'gamma'])
    parser.add_argument('--Exmax', nargs='?', type=float,
                        help="""Maximum value of the excitation energy.
                        Defaults to a reasonable value.""")
    parser.add_argument('--Exmin', nargs='?', type=float,
                        help="""Minimum value of the excitation energy.
                        Defaults to 0.""", default=0.0)
    parser.add_argument('--binwidth', nargs='?', type=float,
                        help="""The bin width to use. Defaults to 0.2 MeV""")
    args = parser.parse_args()

    # Handle summary file
    if len(args.summary_file) < 1:
        raise RuntimeFailure("Found no summary files and none were supplied.")
    if len(args.summary_file) > 1:
        raise RuntimeFailure("Too many summary files were found")
    args.summary_file = args.summary_file[0]

    # Handle parity
    if args.pi is None:
        π = (-1, 1)
    else:
        if len(args.pi) > 2:
            raise RuntimeFailure(f"Unexpected values for parity: {args.pi}")
        π = []
        for pi in args.pi:
            if pi in {'0', '-1', '-'}:
                π.append(-1)
            elif pi in {'1', '+1', '+'}:
                π.append(1)
            elif pi in {'01', '10', '+-', '-+'}:
                π.append(1)
                π.append(-1)
            else:
                raise RuntimeFailure(f"Unexpected values for parity: {args.pi}")
    args.pi = tuple(set(π))

    return args


if __name__ == '__main__':
    args = parse_arguments()
    analyzer = Analyzer(args.summary_file,
                        bin_width=args.binwidth,
                        Ex_max=args.Exmax,
                        Ex_min=args.Exmin)

    if 'scheme' in args.actions:
        analyzer.plot_scheme()
    if 'ld' in args.actions:
        analyzer.plot_total_level_density()
    if 'g' in args.actions or 'gamma' in args.actions:
        analyzer.plot_gamma_strength_function(J=args.j, pi=args.pi)
    if 'ldplay' in args.actions:
        analyzer.plot_level_density()

    # A = Analyzer("summary_Zn70_jun45_2.txt")
    # A = Analyzer("calculations/summary_Ni70_jun45.txt")
    # A = Analyzer("/home/erdos/KSHELL_runs/68Co_ca48mh1_1/summary_Co68_ca48mh1.txt")
    # A.plot_scheme()
    # A.plot_level_density()
    # A.plot_total_level_density(bin_width=0.2, Ex_max=5)

    # Jpi_list = [[0+1,+1],[2+1,+1],[4+1,+1],[6+1,+1],[8+1,+1],[10+1,+1],[12+1,+1],[14+1,+1],[16+1,+1],[18+1,+1],[20+1,+1],[22+1,+1],[24+1,+1],[26+1,+1],[28+1,+1],
    #             [0+1,-1],[2+1,-1],[4+1,-1],[6+1,-1],[8+1,-1],[10+1,-1],[12+1,-1],[14+1,-1],[16+1,-1],[18+1,-1],[20+1,-1],[22+1,-1],[24+1,-1],[26+1,-1],[28+1,-1]]
    # Jpi = list(product(range(0, 30, 2), (1, -1)))
    # J = [0+10,0-10,1+10,1-10,2+10,2-10,3+10,3-10,4+10,4-10,5+10,5-10,6+10,6-10,7+10,7-10,8+10,8-10,9+10,9-10,10+10,10-10]
    # Jpi = [[4, +1]]
    # print(Jpi)
    # A.plot_gamma_strength_function(J=range(0, 29, 2), Ex_max=20, pi=[-1, 1])
    # A.plot_gamma_strength_function(J=J, Ex_max=5)
