# settings.py

# Panel configuration
SECTIONS_PER_PANEL = 3
OUTGOINGS_PER_SECTION = 10
SPARE_OUTGOINGS_PER_SECTION = 1

# Limitations
MAX_CIRCUIT_CB_CURRENT = 20.0
MAX_DISTANCE_TO_PANEL = 150 # in meters
MAX_DISTANCE_BETWEEN_CIRCUITS = 50  # in meters

# Files
INPUT_FILE = 'input.xlsx'
OUTPUT_FILE_XLSX = 'Grouping.xlsx'
OUTPUT_FILE_DXF = 'Grouping.dxf'

#Dxf settings
PANEL_BLOCK_SIZE = 10000
CIRCUIT_BLOCK_SIZE = 5000
BLOCK_TEXT_SIZE = 2000
