# config.py
# Global variables

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2020. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


import argparse
import copy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


MODE_TRAIN = 'TRAIN'
MODE_TEST = 'TEST'
MODE_CV = 'CV'
MODE_PARTITION = 'PARTITION'
MODE_ANNOTATE = 'ANNOTATE'
MODE_WRITE = 'WRITE'
MODE_LOO = 'LOO'
MODES = list([MODE_TRAIN,
              MODE_TEST,
              MODE_CV,
              MODE_PARTITION,
              MODE_ANNOTATE,
              MODE_WRITE,
              MODE_LOO])


class Config:

    def __init__(self):
        """ Constructor
        """
        self.activitynames = [str(i) for i in range(1, 25)]  # Activities 1-24
        self.current_seconds_of_day = 0
        self.current_timestamp = 0
        self.day_of_week = 0
        self.dominant = 0
        self.sensornames = ['D001', 'D003', 'D005', 'D006', 'D007', 'D009', 'D010', 'D011',
                            'D012', 'D013', 'D014', 'D015', 'D017', 'D018', 'E002', 'I001',
                            'I002', 'I006', 'I010', 'I011', 'I012', 'L001', 'L003', 'L005',
                            'L006', 'L007', 'L008', 'LL005', 'LL006', 'LL007', 'LS001', 'LS002',
                            'LS003', 'LS004', 'LS005', 'LS006', 'LS007', 'LS008', 'LS009', 'LS010',
                            'LS011', 'LS012', 'LS013', 'LS014', 'LS015', 'LS016', 'LS017', 'LS018',
                            'LS019', 'LS021', 'LS022', 'LS023', 'LS024', 'LS025', 'LS026', 'LS027',
                            'LS028', 'LS029', 'LS030', 'LS031', 'LS032', 'LS033', 'LS034', 'LS035',
                            'LS036', 'LS037', 'LS039', 'LS042', 'LS043', 'LS044', 'LS045', 'LS046',
                            'LS047', 'LS048', 'LS049', 'LS050', 'LS051', 'LS201', 'LS202', 'LS203',
                            'LS204', 'LS205', 'LS206', 'LS207', 'M001', 'M002', 'M003', 'M004',
                            'M005', 'M006', 'M007', 'M008', 'M009', 'M010', 'M011', 'M012',
                            'M013', 'M014', 'M015', 'M016', 'M017', 'M018', 'M019', 'M020',
                            'M021', 'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028',
                            'M029', 'M030', 'M031', 'M032', 'M033', 'M034', 'M035', 'M036',
                            'M037', 'M038', 'M039', 'M040', 'M041', 'M042', 'M043', 'M044',
                            'M045', 'M046', 'M047', 'M048', 'M049', 'M050', 'M051', 'M052',
                            'MA201', 'MA202', 'MA203', 'MA207', 'P001', 'P002', 'system',
                            'T001', 'T002', 'T003', 'T004', 'T005', 'T101', 'T102', 'T104',
                            'T106', 'T107', 'T108', 'T111']
        self.sensortimes = []
        self.data = []
        self.dstype = []
        self.labels = []
        self.numwin = 0
        self.wincnt = 0
        self.data_filename = "data.txt"
        self.filter_other = True  # Do not consider other in performance
        self.ignore_other = False
        self.cluster_other = False
        self.no_overlap = False  # Do not allow windows to overlap
        self.confusion_matrix = True
        self.mode = MODE_PARTITION  # TRAIN, TEST, CV, PARTITION, ANNOTATE, WRITE, LOO
        self.model_path = "./model/"
        self.model_name = "model"
        self.num_activities = 0
        self.num_clusters = 10
        self.num_sensors = 0
        self.num_set_features = 14
        self.features = 0
        self.num_features = 0
        self.seconds_in_a_day = 86400
        self.max_window = 5  # Reduced from 30 to handle sparse activities
        self.add_pca = False  # Add principal components to feature vector
        self.weightinc = 0.01
        self.windata = np.zeros((self.max_window, 3), dtype=int)
        self.clf = RandomForestClassifier(n_estimators=80,
                                          max_features='sqrt',
                                          bootstrap=True,
                                          criterion="entropy",
                                          min_samples_split=20,
                                          max_depth=None,
                                          n_jobs=4,
                                          class_weight='balanced')

    def set_parameters(self):
        """ Set parameters according to command-line args list.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode',
                            dest='mode',
                            type=str,
                            choices=MODES,
                            default=self.mode,
                            help=('Define the core mode that we will run in, default={}.'
                                  .format(self.mode)))
        parser.add_argument('--data',
                            dest='data',
                            type=str,
                            default=self.data_filename,
                            help=('Data file of sensor data, default={}'
                                  .format(self.data_filename)))
        parser.add_argument('--model',
                            dest='model',
                            type=str,
                            default=self.model_name,
                            help=('Specifies the name of the model to use, default={}'
                                  .format(self.model_name)))
        parser.add_argument('--ignoreother',
                            dest='ignoreother',
                            default=self.ignore_other,
                            action='store_true',
                            help=('Ignores all sensor events affiliated with activity '
                                  'Other_Activity, default={}'.format(self.ignore_other)))
        parser.add_argument('--clusterother',
                            dest='clusterother',
                            default=self.cluster_other,
                            action='store_true',
                            help=('Divides the Other_Activity category into subclasses using '
                                  'k-means clustering.  When activated this sets --ignoreother '
                                  'to False, default={}'.format(self.cluster_other)))
        parser.add_argument('--sensors',
                            dest='sensors',
                            type=str,
                            default=','.join(self.sensornames),
                            help=('Comma separated list of sensors that appear in the data file, '
                                  'default={}'.format(','.join(self.sensornames))))
        parser.add_argument('--activities',
                            dest='activities',
                            type=str,
                            default=','.join(self.activitynames),
                            help=('Comma separated list of activities to use, '
                                  'default={}'.format(','.join(self.activitynames))))
        parser.add_argument('files',
                            metavar='FILE',
                            type=str,
                            nargs='*',
                            help='Data files for AL to process.')
        args = parser.parse_args()

        self.mode = args.mode
        self.data_filename = args.data
        self.model_name = args.model
        self.ignore_other = args.ignoreother
        self.cluster_other = args.clusterother
        if self.cluster_other:
            self.ignore_other = False
        self.sensornames = str(args.sensors).split(',')
        self.num_sensors = len(self.sensornames)
        for i in range(self.num_sensors):
            self.sensortimes.append(0)
            self.dstype.append('n')
        self.activitynames = str(args.activities).split(',')
        self.num_activities = len(self.activitynames)
        files = list()
        if len(args.files) > 0:
            files = copy.deepcopy(args.files)
        else:
            files.append(self.data_filename)

        return files
