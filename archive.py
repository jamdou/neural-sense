"""
A class to handle archiving inputs and results, as well as an :class:`Enum` to help organise profiling results.
"""

import numpy as np
import time as tm
import datetime as dtm
import h5py
import os
import glob
import shutil
from enum import Enum, auto

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

class Archive:
    """
    A class containing a hdf5 archive file using :mod:`h5py`, as well as other path information for organising saved plots, and cuda profiles using nvprof.

    Attributes
    ----------
    executionTimeString : `string`
        The time when the code was first executed, in YYYYmmddTHHMMSS format.
    descriptionOfTest : `string`
        A note of the current aim of the test, to be saved to the hdf5 archive.
    archiveFile : :class:`h5py.File`
        The hdf5 file to use when archiving.
    archivePath : `string`
        The archive path to save the hdf5 archive to.
    plotPath : `string`
        The archive path to save plots to.
    profilePath : `string`
        The archive path to save profile outputs to.
    profileState : :class:`ProfileState`
        A description of which type of profiling is being done now. Used to organise profile output files.
    sourcePath : `string`
        Path to source files.
    profileLocalPath : `string`
        Path to temporary profile outputs, to be properly archived when `profileState` is :obj:`ProfileState.ARCHIVE`.
    """
    def __init__(self, archivePath, descriptionOfTest, profileState = ProfileState.NONE):
        """
        Parameters
        ----------
        archivePath : `string`
            The path where hdf5 archive files are saved. A new directory for the day, then time, will be made here to organise the archives.
        descriptionOfTest : `string`
            A note of the current aim of the test, to be saved to the hdf5 archive.
        profileState : :class:`ProfileState`, optional
            A description of which type of profiling is being done now. Used to organise profile output files. Defaults to :obj:`ProfileState.NONE`.
        """
        # Set up profile directories
        self.profileState = profileState
        self.sourcePath = ".\\"
        self.profileLocalPath = self.sourcePath + "profile\\"
        if profileState == ProfileState.ARCHIVE:
            profileFlagFile = open(self.profileLocalPath + "profileFlag", "r")
            self.executionTimeString = profileFlagFile.read()
            profileFlagFile.close()
            os.remove(self.profileLocalPath + "profileFlag")
        else:
            self.executionTimeString = dtm.datetime.now().strftime("%Y%m%dT%H%M%S")
        
        # Set up archive directories
        self.archivePath = archivePath + self.executionTimeString[:8] + "\\" + self.executionTimeString + "\\"
        self.profilePath = self.archivePath + "profile\\"
        self.plotPath = self.archivePath + "plots\\"
        self.descriptionOfTest = descriptionOfTest
        self.archiveFile = None

        if profileState == ProfileState.ARCHIVE:
            self.archiveFile = h5py.File(self.archivePath + "archive.hdf5", "a")
            self.writeProfiles()
        else:
            if not os.path.exists(self.plotPath):
                os.makedirs(self.plotPath)
            if profileState == ProfileState.TIME_LINE:
                if not os.path.exists(self.profilePath):
                    os.makedirs(self.profilePath)

    def newArchiveFile(self):
        """
        Makes a new hdf5 file at the archive path.
        """
        self.archiveFile = h5py.File(self.archivePath + "archive.hdf5", "w")

    def openArchiveFile(self, oldExecutionTimeString):
        """
        Reads an existing hdf5 archive file, referenced by a given time.

        Parameters
        ----------
        oldExecutionTimeString : `string`
            The time identity of the archive to be opened.
        """
        self.archiveFile = h5py.File(self.archivePath[:-25] + oldExecutionTimeString[:8] + "\\" + oldExecutionTimeString + "\\archive.hdf5", "r")

    def closeArchiveFile(self, doSaveSource = True):
        """
        Archive source code and profile files.

        Parameters
        ----------
        doSaveSource : `boolean`, optional
            If `True`, will write copies of the source code to the archive on closing. Defaults to `True`.
        """
        if self.archiveFile:
            if doSaveSource:
                self.writeCodeSource()
            self.archiveFile.close()
            if self.profileState == ProfileState.TIME_LINE:
                profileFlagFile = open(self.profileLocalPath + "profileFlag", "w")
                profileFlagFile.write(self.executionTimeString)
                profileFlagFile.close()
        else:
            raise Warning("No archive currently open.")

    def writeCodeSource(self):
        """
        Save source code to the hdf5 file.
        """
        archiveGroupCodeSource = self.archiveFile.require_group("codeSource" + self.executionTimeString)
        archiveGroupCodeSource["descriptionOfTest"] = np.asarray([self.descriptionOfTest], dtype = "|S512")
        for codeSourceName in glob.glob(self.sourcePath + "*.py") + glob.glob(self.sourcePath + "*.bat"):
            codeSource = open(codeSourceName, "r")
            archiveGroupCodeSource[codeSourceName.replace(self.sourcePath, "")] = codeSource.read()
            codeSource.close()

    def writeProfiles(self):
        """
        Organise the generated profile files to the `profilePath`.
        """
        for profileName in glob.glob(self.profileLocalPath + "*.prof"):
            shutil.copyfile(profileName, profileName.replace(self.profileLocalPath, self.profilePath))