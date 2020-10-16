import numpy as np
import time as tm
import datetime as dtm
import h5py
import os
import glob
import shutil

class Archive:
    def __init__(self, archivePath, descriptionOfTest, profileState = "None"):
        # Set up profile directories
        self.profileState = profileState
        self.sourcePath = ".\\"
        self.profileLocalPath = self.sourcePath + "profile\\"
        if profileState == "Archive":
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

        if profileState == "Archive":
            self.archiveFile = h5py.File(self.archivePath + self.executionTimeString + "archive.hdf5", "a")
            self.writeProfiles()
        else:
            if not os.path.exists(self.plotPath):
                os.makedirs(self.plotPath)
            if profileState == "TimeLine":
                if not os.path.exists(self.profilePath):
                    os.makedirs(self.profilePath)

    def newArchiveFile(self):
        """
        Make a new hdf5 file at the archive path.
        """
        self.archiveFile = h5py.File(self.archivePath + "archive.hdf5", "w")

    def openArchiveFile(self, oldExecutionTimeString):
        self.archiveFile = h5py.File(self.archivePath[:-25] + oldExecutionTimeString[:8] + "\\" + oldExecutionTimeString + "\\archive.hdf5", "r")

    def closeArchiveFile(self, doSaveSource = True):
        """
        Archive source code and profile files.
        """
        if self.archiveFile:
            if doSaveSource:
                self.writeCodeSource()
            self.archiveFile.close()
            if self.profileState == "TimeLine":
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
        Organise the generated profile files.
        """
        for profileName in glob.glob(self.profileLocalPath + "*.prof"):
            shutil.copyfile(profileName, profileName.replace(self.profileLocalPath, self.profilePath))