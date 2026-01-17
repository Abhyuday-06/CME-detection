import cdflib
import os

# Path to the CDF file
cdf_file_path = 'E:\\Programming\\CME-Detection\\data\\SWIS-ISSDC\\positive\\AL1_ASW91_L2_BLK_20240509_UNP_9999_999999_V02.cdf'

try:
    # Open the CDF file
    cdf_file = cdflib.CDF(cdf_file_path)

    # Print the file information
    print("CDF File Info:")
    print(cdf_file.cdf_info())

    # List all variables
    print("\nVariables:")
    variables = cdf_file.cdf_info().zVariables
    print(variables)

    print("\n--- Variable Details for SQL Schema ---")
    for var_name in variables:
        var_info = cdf_file.varinq(var_name)
        var_atts = cdf_file.varattsget(var_name)
        
        # Handle VDRInfo object (access via attributes)
        data_type = var_info.Data_Type
        dim_sizes = var_info.Dim_Sizes
        
        print(f"Variable: {var_name}")
        print(f"  CDF Type Code: {data_type}")
        print(f"  Shape: {dim_sizes}")
        print(f"  Description: {var_atts.get('CATDESC', 'N/A')}")
        print(f"  Units: {var_atts.get('UNITS', 'N/A')}")
        print("-" * 30)

except Exception as e:
    print(f"Error reading CDF file: {e}")
