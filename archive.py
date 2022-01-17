"""
A class to handle archiving inputs and results, as well as an :class:`enum.Enum` to help organise profiling results.
"""

import numpy as np
import matplotlib.pyplot as plt
import time as tm
import datetime as dtm
import h5py
import os
import glob
import shutil
import sys, getopt
from enum import Enum, auto
# from sim.manager import SimulationManager
from test_signal import TestSignal

class ProfileState(Enum):
    """
    A description of which type of profiling is being done. Used to organise profile output files.
    """

    NONE = auto()
    """
    No profiling is being done.
    """

    TIME_LINE = auto()
    """
    Profiling is being done on timing.
    """

    METRIC = auto()
    """
    Profiling is being done on general kernel behaviour.
    """

    INSTRUCTION_LEVEL = auto()
    """
    Profiling is being done on a per instruction basis.
    """

    ARCHIVE = auto()
    """
    The results of past profiling are to be moved to a single location to be archived.
    """

def handle_arguments():
    """
    Reads and handles arguments (mainly to do with profiling) from the command line.

    Returns
    -------
    archive_path : :obj:`str`
        The path where hdf5 archive files are saved. A new directory for the day, then time, will be made here to organise the archives.
    profile_state : :class:`ProfileState`, optional
        A description of which type of profiling is being done now. Used to organise profile output files. Defaults to :obj:`ProfileState.NONE`.
    description_of_test : :obj:`str`
        A note of the current aim of the test, to be saved to the hdf5 archive.
    """
    help_message = """
    \033[36m-h  --help      \033[33mShow this message
    \033[36m-a  --archive   \033[33mSelect an alternate archive path.
                    \033[32mDefault:
                        \033[36m.\\archive\033[0m
    \033[36m-p  --profile   \033[33mSelect what type of nvprof profiling to be
                    done, from:
                        \033[36mnone \033[32m(default)  \033[33mRun normally
                        \033[36mtimeline            \033[33mSave timeline
                        \033[36mmetric              \033[33mSave metrics
                        \033[36minstructionlevel    \033[33mSave per instruction
                                            metrics
                        \033[36marchive             \033[33mArchive results,
                                            don't run anything
                                            else
                    \033[35mOnly used for automation with profiling, if
                    you're not doing this, then leave this blank.\033[0m
    """

    # Command line arguments. Probably don't worry too much about these. Mostly used for profiling.
    profile_state = ProfileState.NONE
    archive_path = ".\\archive\\"
    options, arguments = getopt.getopt(sys.argv[1:], "hpa", ["help", "profile=", "archive="])
    for option, argument in options:
        if option in ("--help", "-h"):
            print(help_message)
            exit()
        elif option in ("--profile", "-p"):
            if argument == "timeline":
                profile_state = ProfileState.TIME_LINE
            elif argument == "metric":
                profile_state = ProfileState.METRIC
            elif argument == "instructionlevel":
                profile_state = ProfileState.INSTRUCTION_LEVEL
            elif argument == "archive":
                profile_state = ProfileState.ARCHIVE
        elif option in ("--archive", "-a"):
            archive_path = argument + "\\"

    return profile_state, archive_path

class Archive:
    """
    A class containing a hdf5 archive file using :mod:`h5py`, as well as other path information for organising saved plots, and cuda profiles using nvprof.

    Attributes
    ----------
    execution_time_string : :obj:`str`
        The time when the code was first executed, in YYYYmmddTHHMMSS format.
    description_of_test : :obj:`str`
        A note of the current aim of the test, to be saved to the hdf5 archive.
    archive_file : :class:`h5py.File`
        The hdf5 file to use when archiving.
    archive_path : :obj:`str`
        The archive path to save the hdf5 archive to.
    plot_path : :obj:`str`
        The archive path to save plots to.
    profile_path : :obj:`str`
        The archive path to save profile outputs to.
    profile_state : :class:`ProfileState`
        A description of which type of profiling is being done now. Used to organise profile output files.
    source_path : :obj:`str`
        Path to source files.
    profile_local_path : :obj:`str`
        Path to temporary profile outputs, to be properly archived when `profile_state` is :obj:`ProfileState.ARCHIVE`.
    """
    def __init__(self, archive_path, description_of_test, profile_state = ProfileState.NONE):
        """
        Parameters
        ----------
        archive_path : :obj:`str`
            The path where hdf5 archive files are saved. A new directory for the day, then time, will be made here to organise the archives.
        description_of_test : :obj:`str`
            A note of the current aim of the test, to be saved to the hdf5 archive.
        profile_state : :class:`ProfileState`, optional
            A description of which type of profiling is being done now. Used to organise profile output files. Defaults to :obj:`ProfileState.NONE`.
        """
        # Set up profile directories
        self.profile_state = profile_state
        self.source_path = ".\\"
        self.profile_local_path = f"{self.source_path}profile\\"
        if profile_state == ProfileState.ARCHIVE:
            profile_flag_file = open(f"{self.profile_local_path}profile_flag", "r")
            self.execution_time_string = profile_flag_file.read()
            profile_flag_file.close()
            os.remove(f"{self.profile_local_path}profile_flag")
        else:
            self.execution_time_string = dtm.datetime.now().strftime("%Y%m%dT%H%M%S")
        
        # Set up archive directories
        self.archive_path = f"{archive_path}{self.execution_time_string[:8]}\\{self.execution_time_string}\\"
        self.profile_path = f"{self.archive_path}profile\\"
        self.plot_path = f"{self.archive_path}plots\\"
        self.description_of_test = description_of_test
        self.archive_file = None

        if profile_state == ProfileState.ARCHIVE:
            self.archive_file = h5py.File(f"{self.archive_path}archive.hdf5", "a")
            self.write_profiles()
        else:
            if not os.path.exists(self.plot_path):
                os.makedirs(self.plot_path)
            if profile_state == ProfileState.TIME_LINE:
                if not os.path.exists(self.profile_path):
                    os.makedirs(self.profile_path)

    def new_archive_file(self):
        """
        Makes a new hdf5 file at the archive path.
        """
        self.archive_file = h5py.File(f"{self.archive_path}archive.hdf5", "w")

    def open_archive_file(self, old_execution_time_string):
        """
        Reads an existing hdf5 archive file, referenced by a given time.

        Parameters
        ----------
        old_execution_time_string : :obj:`str`
            The time identity of the archive to be opened.
        """
        self.archive_file = h5py.File(f"{self.archive_path[:-25]}{old_execution_time_string[:8]}\\{old_execution_time_string}\\archive.hdf5", "r")

    def close_archive_file(self, do_save_source = True):
        """
        Archive source code and profile files.

        Parameters
        ----------
        do_save_source : :obj:`bool`, optional
            If :obj:`True`, will write copies of the source code to the archive on closing. Defaults to :obj:`True`.
        """
        if self.archive_file:
            if do_save_source:
                self.write_code_source()
            self.archive_file.close()
            if self.profile_state == ProfileState.TIME_LINE:
                profile_flag_file = open(f"{self.profile_local_path}profile_flag", "w")
                profile_flag_file.write(self.execution_time_string)
                profile_flag_file.close()
        else:
            raise Warning("No archive currently open.")

    def write_code_source(self):
        """
        Save source code to the hdf5 file.
        """
        archive_group_code_source = self.archive_file.require_group(f"code_source{self.execution_time_string}")
        archive_group_code_source["description_of_test"] = np.asarray([self.description_of_test], dtype = "|S512")
        for code_source_name in glob.glob(f"{self.source_path}**\\*.py", recursive = True) + glob.glob(f"{self.source_path}**\\*.bat", recursive = True):
            code_source = open(code_source_name, "r")
            archive_group_code_source[code_source_name.replace(self.source_path, "")] = code_source.read()
            code_source.close()

    def write_profiles(self):
        """
        Organise the generated profile files to the `profile_path`.
        """
        for profile_name in glob.glob(f"{self.profile_local_path}*"):
            shutil.copyfile(profile_name, profile_name.replace(self.profile_local_path, self.profile_path))

    def write_plot(self, title:str, file_name:str):
        """
        Writes plots to the archive file path in both png and pdf formats both with and without time labels.

        Parameters
        ----------
        title : :obj:`str`
            The title to be drawn on the plot.
        file_name : :obj:`str`
            The path that the plot is written to.
        """
        archive_index_text = ""
        if os.path.isfile(f"{self.plot_path}{file_name}.png"):
            archive_index = 1
            while os.path.isfile(f"{self.plot_path}{file_name}.png"):
                archive_index += 1
            archive_index_text = f"_{archive_index}"

        plt.title(title)
        plt.savefig(f"{self.plot_path}{file_name}{archive_index_text}_publication.pdf")
        plt.savefig(f"{self.plot_path}{file_name}{archive_index_text}_publication.png")

        plt.title(f"{self.execution_time_string}\n{title}")
        plt.savefig(f"{self.plot_path}{file_name}{archive_index_text}.pdf")
        plt.savefig(f"{self.plot_path}{file_name}{archive_index_text}.png")

class ExperimentResults:
    """
    A class that contains information from the results of a complete experiment, whether it was done via simulations, or in the lab.

    Attributes
    ----------
    frequency : :class:`numpy.ndarray` of :class:`numpy.float64`
        The dressing Rabi frequencies at which the experiments were run. In units of Hz.
    frequency_amplitude : :class:`numpy.ndarray` of :class:`numpy.float64`
        The Fourier sine coefficient of the signal being measured, as determined by the experiment. In units of Hz.
    """
    def __init__(self, frequency:np.ndarray = None, frequency_amplitude:np.ndarray = None, archive_time:str = None, experiment_type:str = "unknown"):
        """
        Parameters
        ----------
        frequency : :class:`numpy.ndarray` of :class:`numpy.float64`, optional
            The dressing Rabi frequencies at which the experiments were run. In units of Hz.
        frequency_amplitude : :class:`numpy.ndarray` of :class:`numpy.float64`, optional
            The Fourier sine coefficient of the signal being measured, as determined by the experiment. In units of Hz.
        archive_time : :class:`numpy.ndarray` of :class:`str`, optional
            The time the original archive was taken from
        """
        self.frequency = frequency
        self.frequency_amplitude = frequency_amplitude
        self.archive_time = archive_time
        self.experiment_type = experiment_type

    @staticmethod
    def new_from_simulation_manager(simulation_manager):
        """
        A constructor that creates a new :class:`ExperimentResults` object from an already evaluated :class:`SimulationManager` object. That is, make an experiment results object based off of simulation results.

        Parameters
        ----------
        simulation_manager : :class:`SimulationManager`
            The simulation manager object to read the results of.

        Returns
        -------
        experiment_results : :class:`ExperimentResults`
            A new object containing the results of `simulation_manager`.
        """
        frequency = 1*simulation_manager.frequency
        frequency_amplitude = 1*simulation_manager.frequency_amplitude[:frequency.size]

        return ExperimentResults(frequency, frequency_amplitude, experiment_type = "simulation")

    def plot(self, archive:Archive = None, test_signal:TestSignal = None):
        """
        Plots the experiment results contained in the object. Optionally saves the plots, as well as compares the results to numerically calculated values.

        Parameters
        ----------
        archive : :class:`archive.Archive`, optional
            If specified, will save the generated plot to the archive's :attr:`archive.Archive.plot_path`.
        test_signal : :class:`test_signal:TestSignal`, optional
            The signal that was being measured during the experiment. If specified, this function will plot the sine Fourier transform of the signal behind the measured coefficients of the experiment results.
        """
        plt.figure()
        if test_signal:
            plt.plot(test_signal.frequency, test_signal.frequency_amplitude, "-k")
        plt.plot(self.frequency, self.frequency_amplitude, "xr")
        if test_signal:
            plt.legend(["Fourier Transform", "Measured"])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (Hz)")
        plt.xlim([0, np.max(self.frequency)])
        # plt.ylim([-0.08, 0.08])
        plt.grid()
        if archive:
            archive.write_plot("Measured Frequency Amplitude", "measured_frequency_amplitude")
        plt.draw()

    def write_to_archive(self, archive:Archive):
        """
        Saves the contents of the experiment results to an archive.

        Parameters
        ----------
        archive : :class:`archive.archive`
            The archive object to save the results to.
        """
        archive_group_experiment_results = archive.archive_file.require_group("experiment_results")
        archive_group_experiment_results["frequency"] = self.frequency
        archive_group_experiment_results["frequency_amplitude"] = self.frequency_amplitude
        if self.archive_time is not None:
            archive_group_experiment_results.attrs["archive_time"] = self.archive_time
        if self.experiment_type is not None:
            archive_group_experiment_results.attrs["experiment_type"] = self.experiment_type
    
    @staticmethod
    def new_from_archive_time(archive:Archive, archive_time):
        archive_previous = Archive(archive.archive_path[:-25], "")
        archive_previous.open_archive_file(archive_time)
        archive_group_experiment_results = archive_previous.archive_file.require_group("experiment_results")

        frequency = np.asarray(archive_group_experiment_results["frequency"])
        frequency_amplitude = np.asarray(archive_group_experiment_results["frequency_amplitude"])
        if "archive_time" in archive_group_experiment_results.attrs:
            archive_time_real = archive_group_experiment_results.attrs["archive_time"]
        else:
            archive_time_real = archive_time
        if "experiment_type" in archive_group_experiment_results.attrs:
            experiment_type = archive_group_experiment_results.attrs["experiment_type"]
        else:
            experiment_type = "unknown"

        # archive_previous.close_archive_file(False)

        return ExperimentResults(frequency, frequency_amplitude, archive_time = archive_time_real, experiment_type = experiment_type)

class RamseyResults:
    def __init__(self, time:np.ndarray = None, amplitude:np.ndarray = None, archive_time:str = None, experiment_type:str = "unknown") -> None:
        self.time = time.copy()
        self.amplitude = amplitude.copy()
        
        self.archive_time = archive_time
        self.experiment_type = experiment_type

    def plot(self, archive:Archive):
        plt.figure()
        plt.plot(self.time, self.amplitude, "k-")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Hz)")
        if archive:
            archive.write_plot("Ramsey sampling", "ramsey_sampling")
        plt.draw()

    def write_to_archive(self, archive:Archive):
        archive_group_ramsey_results = archive.archive_file.require_group("ramsey_results")
        archive_group_ramsey_results["time"] = self.time
        archive_group_ramsey_results["amplitude"] = self.amplitude
        if self.archive_time is not None:
            archive_group_ramsey_results.attrs["archive_time"] = self.archive_time
        if self.experiment_type is not None:
            archive_group_ramsey_results.attrs["experiment_type"] = self.experiment_type
    
    @staticmethod
    def new_from_archive_time(archive:Archive, archive_time):
        archive_previous = Archive(archive.archive_path[:-25], "")
        archive_previous.open_archive_file(archive_time)
        archive_group_ramsey_results = archive_previous.archive_file.require_group("ramsey_results")

        time = np.asarray(archive_group_ramsey_results["time"])
        amplitude = np.asarray(archive_group_ramsey_results["amplitude"])
        if "archive_time" in archive_group_ramsey_results.attrs:
            archive_time_real = archive_group_ramsey_results.attrs["archive_time"]
        else:
            archive_time_real = archive_time
        if "experiment_type" in archive_group_ramsey_results.attrs:
            experiment_type = archive_group_ramsey_results.attrs["experiment_type"]
        else:
            experiment_type = "unknown"

        return RamseyResults(time, amplitude, archive_time = archive_time_real, experiment_type = experiment_type)