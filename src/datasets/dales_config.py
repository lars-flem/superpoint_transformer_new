import numpy as np


########################################################################
#                         Download information                         #
########################################################################

FORM_URL = 'https://docs.google.com/forms/d/e/1FAIpQLSefhHMMvN0Uwjnj_vWQgYSvtFOtaoGFWsTIcRuBTnP09NHR7A/viewform?fbzx=5530674395784263977'

# DALES in LAS format
LAS_TAR_NAME = 'dales_semantic_segmentation_las.tar.gz'
LAS_UNTAR_NAME = "dales_las"

# DALES in PLY format
PLY_TAR_NAME = 'dales_semantic_segmentation_ply.tar.gz'
PLY_UNTAR_NAME = "dales_ply"

# DALES in PLY, only version with intensity and instance labels
OBJECTS_TAR_NAME = 'DALESObjects.tar.gz'
OBJECTS_UNTAR_NAME = "DALESObjects"


########################################################################
#                              Data splits                             #
########################################################################

# The validation set was arbitrarily chosen as the x last train tiles:
TILES = {
    'train': [
        '5080_54435',
        '5190_54400',
        '5105_54460',
        '5130_54355',
        '5165_54395',
        '5185_54390',
        '5180_54435',
        '5085_54320',
        '5100_54495',
        '5110_54320',
        '5140_54445',
        '5105_54405',
        '5185_54485',
        '5165_54390',
        '5145_54460',
        '5110_54460',
        '5180_54485',
        '5150_54340',
        '5145_54405',
        '5145_54470',
        '5160_54330',
        '5135_54495',
        '5145_54480',
        '5115_54480',
        '5110_54495',
        '5095_54440'],

    'val': [
        '5145_54340',
        '5095_54455',
        '5110_54475'],

    'test': [
        '5080_54470',
        '5100_54440',
        '5140_54390',
        '5080_54400',
        '5155_54335',
        '5150_54325',
        '5120_54445',
        '5135_54435',
        '5175_54395',
        '5100_54490',
        '5135_54430']}


########################################################################
#                                Labels                                #
########################################################################


#Original DALES classes:

"""
DALES_NUM_CLASSES = 8

ID2TRAINID = np.asarray([8, 0, 1, 2, 3, 4, 5, 6, 7])

CLASS_NAMES = [
    'Ground',
    'Vegetation',
    'Cars',
    'Trucks',
    'Power lines',
    'Fences',
    'Poles',
    'Buildings',
    'Unknown']

CLASS_COLORS = np.asarray([
    [243, 214, 171],  # sunset
    [ 70, 115,  66],  # fern green
    [233,  50, 239],
    [243, 238,   0],
    [190, 153, 153],
    [  0, 233,  11],
    [239, 114,   0],
    [214,   66,  54],  # vermillon
    [  0,   8, 116]])

# For instance segmentation
MIN_OBJECT_SIZE = 100
THING_CLASSES = [2, 3, 4, 5, 6, 7]
STUFF_CLASSES = [i for i in range(DALES_NUM_CLASSES) if not i in THING_CLASSES]

"""

#Binary DALES classes (ground vs not_ground)

DALES_NUM_CLASSES = 2

ID2TRAINID = np.asarray([8, 0, 1, 2, 3, 4, 5, 6, 7])


ID2TRAINID[0] = 0                  # Ground -> ground
ID2TRAINID[1] = 1                  # Vegetation -> not_ground
ID2TRAINID[2] = 1                  # Cars -> not_ground
ID2TRAINID[3] = 1                  # Trucks -> not_ground
ID2TRAINID[4] = 1                  # Power lines -> not_ground
ID2TRAINID[5] = 1                  # Fences -> not_ground
ID2TRAINID[6] = 1                  # Poles -> not_ground
ID2TRAINID[7] = 1                  # Buildings -> not_ground
ID2TRAINID[8] = 2                  # Unknown -> not_ground



CLASS_NAMES = [
    'Ground',
    'Not Ground',
    'Ignored']

CLASS_COLORS = np.asarray([
    [243, 214, 171],  # sunset
    [ 70, 115,  66],  # fern green
    [  0,   8, 116]])

MIN_OBJECT_SIZE = 100
THING_CLASSES = [2, 3, 4, 5, 6, 7]
STUFF_CLASSES = [i for i in range(DALES_NUM_CLASSES) if not i in THING_CLASSES]