import numpy as np
import os, sys

MRC_2014_SPEC_TABLE = """
+---------+--------+----------+--------+-------------------------------+
|Long word|Byte    |Data type |Name    |Description                    |
+=========+========+==========+========+===============================+
|1        |1-4     |Int32     |NX      |Number of columns              |
+---------+--------+----------+--------+-------------------------------+
|2        |5-8     |Int32     |NY      |Number of rows                 |
+---------+--------+----------+--------+-------------------------------+
|3        |9-12    |Int32     |NZ      |Number of sections             |
+---------+--------+----------+--------+-------------------------------+
|4        |13-16   |Int32     |MODE    |Data type                      |
+---------+--------+----------+--------+-------------------------------+
|8        |29-32   |Int32     |MX      |Number of intervals along X of |
|         |        |          |        |the "unit cell"                |
+---------+--------+----------+--------+-------------------------------+
|9        |33-36   |Int32     |MY      |Number of intervals along Y of |
|         |        |          |        |the "unit cell"                |
+---------+--------+----------+--------+-------------------------------+
|10       |37-40   |Int32     |MZ      |Number of intervals along Z of |
|         |        |          |        |the "unit cell"                |
+---------+--------+----------+--------+-------------------------------+
|11-13    |41-52   |Float32   |CELLA   |Cell dimension in angstroms    |
|         |        |          |        |(whole volume)                 |
+---------+--------+----------+--------+-------------------------------+
|17       |65-68   |Int32     |MAPC    |axis corresponding to columns  |
|         |        |          |        |(1,2,3 for X,Y,Z)              |
+---------+--------+----------+--------+-------------------------------+
|18       |69-72   |Int32     |MAPR    |axis corresponding to rows     |
|         |        |          |        |(1,2,3 for X,Y,Z)              |
+---------+--------+----------+--------+-------------------------------+
|19       |73-76   |Int32     |MAPS    |axis corresponding to sections |
|         |        |          |        |(1,2,3 for X,Y,Z)              |
+---------+--------+----------+--------+-------------------------------+
|20       |77-80   |Float32   |DMIN    |Minimum density value          |
+---------+--------+----------+--------+-------------------------------+
|21       |81-84   |Float32   |DMAX    |Maximum density value          |
+---------+--------+----------+--------+-------------------------------+
|22       |85-88   |Float32   |DMEAN   |Mean density value             |
+---------+--------+----------+--------+-------------------------------+
|23       |89-92   |Int32     |ISPG    |Space group number 0, 1, or 401|
+---------+--------+----------+--------+-------------------------------+
|24       |93-96   |Int32     |NSYMBT  |Number of bytes in extended    |
|         |        |          |        |header                         |
+---------+--------+----------+--------+-------------------------------+
|27       |105-108 |String    |EXTTYPE |Extended header type           |
+---------+--------+----------+--------+-------------------------------+
|28       |109-112 |Int32     |NVERSION|Format version identification  |
|         |        |          |        |number                         |
+---------+--------+----------+--------+-------------------------------+
|50-52    |197-208 |Int32     |ORIGIN  |Origin in X, Y, Z used in      |
|         |        |          |        |transform                      |
+---------+--------+----------+--------+-------------------------------+
|53       |209-212 |String    |MAP     |Character string 'MAP' to      |
|         |        |          |        |identify file type             |
+---------+--------+----------+--------+-------------------------------+
|54       |213-216 |String    |MACHST  |Machine stamp                  |
+---------+--------+----------+--------+-------------------------------+
|55       |217-220 |Float32   |RMS     |RMS deviation of map from mean |
|         |        |          |        |density                        |
+---------+--------+----------+--------+-------------------------------+
|56       |221-224 |Int32     |NLABL   |Number of labels being used    |
+---------+--------+----------+--------+-------------------------------+
|57-256   |225-1024|String(80)|LABEL   |10 80-character text labels    |
+---------+--------+----------+--------+-------------------------------+
"""

OMRC_2014_SPEC_DICT = OrderedDict([('nx',
              {'description': 'Number of columns',
               'offset': 0,
               'dtype': np.dtype('int32'),
               'value': []}),                                   
             ('ny',
              {'description': 'Number of rows',
               'offset': 4,
               'dtype': np.dtype('int32'),
               'value': []}),                                   
             ('nz',
              {'description': 'Number of sections',
               'offset': 8,
               'dtype': np.dtype('int32'),
               'value': []}),                                   
             ('mode',
              {'description': 'Data type',
               'offset': 12,
               'dtype': np.dtype('int32'),
               'value': []}),                                   
             ('mx',
              {'description': 'Number of intervals along X of the "unit cell"',
               'offset': 28,
               'dtype': np.dtype('int32'),
               'value': []}),                                   
             ('my',
              {'description': 'Number of intervals along Y of the "unit cell"',
               'offset': 32,
               'dtype': np.dtype('int32'),
               'value': []}),                                   
             ('mz',
              {'description': 'Number of intervals along Z of the "unit cell"',
               'offset': 36,
               'dtype': np.dtype('int32'),
               'value': []}),                                   
             ('cella',
              {'description': 'Cell dimension in angstroms (whole volume)',
               'offset': 40,
               'dtype': np.dtype('float32'),
               'value': []}),                                   
             ('mapc',
              {'description': 'axis corresponding to columns (1,2,3 for X,Y,Z)',
               'offset': 64,
               'dtype': np.dtype('int32'),
               'value': []}),                                   
             ('mapr',
              {'description': 'axis corresponding to rows (1,2,3 for X,Y,Z)',
               'offset': 68,
               'dtype': np.dtype('int32'),
               'value': []}),                                   
             ('maps',
              {'description': 'axis corresponding to sections (1,2,3 for X,Y,Z)',
               'offset': 72,
               'dtype': np.dtype('int32'),
               'value': []}),                                   
             ('dmin',
              {'description': 'Minimum density value',
               'offset': 76,
               'dtype': np.dtype('float32'),
               'value': []}),                                   
             ('dmax',
              {'description': 'Maximum density value',
               'offset': 80,
               'dtype': np.dtype('float32'),
               'value': []}),                                   
             ('dmean',
              {'description': 'Mean density value',
               'offset': 84,
               'dtype': np.dtype('float32'),
               'value': []}),                                   
             ('ispg',
              {'description': 'Space group number 0, 1, or 401',
               'offset': 88,
               'dtype': np.dtype('int32'),
               'value': []}),                                   
             ('nsymbt',
              {'description': 'Number of bytes in extended header',
               'offset': 92,
               'dtype': np.dtype('int32'),
               'value': []}),                                   
             ('exttype',
              {'description': 'Extended header type',
               'offset': 104,
               'dtype': np.dtype('S1'),
               'value': []}),                                   
             ('nversion',
              {'description': 'Format version identification number',
               'offset': 108,
               'dtype': np.dtype('int32'),
               'value': []}),
             ('origin',
              {'description': 'Origin in X, Y, Z used in transform',
               'offset': 196,
               'dtype': np.dtype('int32'),
               'value': []}),
             ('map',
              {'description': "Character string 'MAP' to identify file type",
               'offset': 208,
               'dtype': np.dtype('S1'),
               'value': []}),
             ('machst',
              {'description': 'Machine stamp',
               'offset': 212,
               'dtype': np.dtype('S1'),
               'value': []}),
             ('rms',
              {'description': 'RMS deviation of map from mean density',
               'offset': 216,
               'dtype': np.dtype('float32'),
               'value':[]}),
             ('nlabl',
              {'description': 'Number of labels being used',
               'offset': 220,
               'value': []}),
             ('label',
              {'description': '10 80-character text labels',
               'offset': 224,
               'dtype': np.dtype('S1'),
               'value': []})])

# Extended header (first section) for the `FEI1` type
MRC_FEI_EXT_HEADER_SECTION = """
+---------+---------+---------+---------------+------------------------------+
|Long word|Byte     |Data type|Name           |Description                   |
+=========+=========+=========+===============+==============================+
|1        |1025-1028|Float32  |A_TILT         |Alpha tilt, in degrees        |
+---------+---------+---------+---------------+------------------------------+
|2        |1029-1032|Float32  |B_TILT         |Beta tilt, in degrees         |
+---------+---------+---------+---------------+------------------------------+
|3        |1033-1036|Float32  |X_STAGE        |Stage x position. Normally in |
|         |         |         |               |SI units (meters), but some   |
|         |         |         |               |older files may be in         |
|         |         |         |               |micrometers.(values larger    |
|         |         |         |               |than 1)                       |
+---------+---------+---------+---------------+------------------------------+
|4        |1037-1040|Float32  |Y_STAGE        |Stage y position              |
+---------+---------+---------+---------------+------------------------------+
|5        |1041-1044|Float32  |Z_STAGE        |Stage z position              |
+---------+---------+---------+---------------+------------------------------+
|6        |1045-1048|Float32  |X_SHIFT        |Stage x shift. For units see  |
|         |         |         |               |remarks on X_STAGE            |
+---------+---------+---------+---------------+------------------------------+
|7        |1049-1052|Float32  |Y_SHIFT        |Stage y shift                 |
+---------+---------+---------+---------------+------------------------------+
|8        |1053-1056|Float32  |DEFOCUS        |Defocus as read from the      |
|         |         |         |               |microscope. For units see     |
|         |         |         |               |remarks on X_STAGE.           |
+---------+---------+---------+---------------+------------------------------+
|9        |1057-1060|Float32  |EXP_TIME       |Exposure time in seconds      |
+---------+---------+---------+---------------+------------------------------+
|10       |1061-1064|Float32  |MEAN_INT       |Mean value of the image       |
+---------+---------+---------+---------------+------------------------------+
|11       |1065-1068|Float32  |TILT_AXIS      |Orientation of the tilt axis  |
|         |         |         |               |in the image in degrees.      |
|         |         |         |               |Vertical to the top is 0      |
|         |         |         |               |degrees, the direction of     |
|         |         |         |               |positive rotation is          |
|         |         |         |               |anti-clockwise.               |
+---------+---------+---------+---------------+------------------------------+
|12       |1069-1072|Float32  |PIXEL_SIZE     |Pixel size of the images in SI|
|         |         |         |               |units (meters)                |
+---------+---------+---------+---------------+------------------------------+
|13       |1073-1076|Float32  |MAGNIFICATION  |Magnification used for        |
|         |         |         |               |recording the images          |
+---------+---------+---------+---------------+------------------------------+
|14       |1077-1080|Float32  |HT             |Value of the high tension in  |
|         |         |         |               |SI units (volts)              |
+---------+---------+---------+---------------+------------------------------+
|15       |1081-1084|Float32  |BINNING        |The binning of the CCD or STEM|
|         |         |         |               |acquisition                   |
+---------+---------+---------+---------------+------------------------------+
|16       |1085-1088|Float32  |APPLIED_DEFOCUS|The intended application      |
|         |         |         |               |defocus in SI units (meters), |
|         |         |         |               |as defined for example in the |
|         |         |         |               |tomography parameters view    |
+---------+---------+---------+---------------+------------------------------+
"""
MRC_HEADER_SIZE = 1024
MRC_FEI_SECTION_SIZE = 128
MRC_FEI_NUM_SECTIONS = 1024


class MRCreader(object):
    '''
    A reader for mrc files:
    For a comprehensive MRC reader, please use the odl.contrib.mrc module in the odl package.
    Args:
         filename string
         dtype string
    '''
    def __init__(self, filename, data_kind):

        try:
            assert(os.path.exists( os.path.abspath( filename)))
        except AssertionError:
            raise Exception('File does not exist.')

        self.fn = filename
        self.__file = open(self.fn, 'rb', buffering = 0)
        if 'b' not in self.__file.mode:
            raise ValueError("`file` must be opened in binary mode, "
                             "but mode 'is {}'".format(self.file.mode))


        self.kind = data_kind

        self.__header = read_mrc_header(self.__file, )
    @property
    def file(self):
        '''Return the __file object'''
        return self.__file

def read_mrc_header(self):
    header = OrderedDict()
    for field in OMRC_2014_SPEC_DICT:
        

def read_extended_header(filename)

def read_mrc(filename, dtype):
'''
'''


