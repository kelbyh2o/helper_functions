"""
This module contains utilities to work folders, files, and data download from the web
    
Author: Kelby Kramer, Massachusetts Institute of Technology and Woods Hole Oceanographic Institution, 2024
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import netCDF4 as nc4
from bs4 import BeautifulSoup
import time
from datetime import datetime

import requests


###################################################################################################
# FILE HANDLING FUNCTIONS
###################################################################################################


def create_site_file_structure(site_folder, site_name):
    """
    Create the structure of subfolders for each site in a given folder location

    KK MIT/WHOI 2024

    Arguments:
    -----------
    site_folder: str
        folder where the images are to be downloaded
    site_name: str
        name of the site or location

    Returns:
    -----------
    filepaths: list of str
        filepaths of the folders that were created
    """
    filepaths = []

    # Create site folder
    site_path = os.path.join(site_folder, site_name)
    os.makedirs(site_path, exist_ok=True)
    filepaths.append(site_path)

    # Define the structure
    structure = {
        'site_conditions': {
            'buoy_data': ['NDBC', 'CDIP', 'WIS'],
            'hindcast_data': ['WAVEWATCH_III'],
            'tide_data': []
        },
        'satellite_imagery': {
            'visual': ['planet', 'S2', 'landsat'],
            'radar': ['S1']
        },
        'ground_truth': ['profiles', 'sonar'],
        'products': []
    }

    # Create folders according to the structure
    for folder, contents in structure.items():
        folder_path = os.path.join(site_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        filepaths.append(folder_path)
        if isinstance(contents, dict):
            for subfolder, subcontents in contents.items():
                subfolder_path = os.path.join(folder_path, subfolder)
                os.makedirs(subfolder_path, exist_ok=True)
                filepaths.append(subfolder_path)
                for item in subcontents:
                    item_path = os.path.join(subfolder_path, item)
                    os.makedirs(item_path, exist_ok=True)
                    filepaths.append(item_path)
        else:
            for item in contents:
                item_path = os.path.join(folder_path, item)
                os.makedirs(item_path, exist_ok=True)
                filepaths.append(item_path)

    return filepaths

def create_folder_structure_dict(folder_path):
    """
    Create a dictionary of the folder structure of a given site

    KK MIT/WHOI 2024

    Arguments:
    -----------
    folder_path: str
        folder where the images are to be downloaded
    
    Returns:
    -----------
    folder_structure: dict
        dictionary of the folder structure
    """
    folder_structure = {}
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            folder_structure[item] = create_folder_structure_dict(item_path)
        else:
            folder_structure.setdefault("files", []).append(item)
    return folder_structure

def plot_folder_structure(folder_structure, level=0):
    """
    Plot the folder structure of a given site

    KK MIT/WHOI 2024

    Arguments:
    -----------
    folder_structure: dict
        dictionary of the folder structure
    level: int
        level of the folder structure to plot
    
    Returns:
    -----------
    None
    """
    for item, contents in folder_structure.items():
        if item == "files":
            continue
        print("\t" * level + item)
        if isinstance(contents, dict):
            plot_folder_structure(contents, level + 1)
        else:
            print("\t" * (level + 1) + contents)


def open_url(url):
    """
    Open a URL using Beautiful Soup return the content

    KK MIT/WHOI 2024

    Arguments:
    -----------
    url: str
        URL to open

    Returns:
    -----------
    soup: BeautifulSoup object
        content of the URL
    """
    # Check the URL is accessible
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException as e:
        print(f"Failed to access the URL: {url}")
        print(e)
        return None
    # Parse the content using Beautiful Soup
    content = response.text
    soup = BeautifulSoup(content, 'html.parser')
    return soup
            

###################################################################################################
# DATA DOWNLOAD FUNCTIONS
###################################################################################################

def download_NDBC_data(buoy_id, years, folder_path, new_folder, dl_stdmet, dl_cwind, dl_swden, dl_swdir, dl_swdir2, dl_swr1, dl_swr2):
    """
    Download data from the National Data Buoy Center (NDBC) website for a given buoy and years

    KK MIT/WHOI 2024

    Arguments:
    -----------
    buoy_id: int
        ID of the buoy to download data from
    years: list of int
        years to download data from
    folder_path: str
        folder where the data is to be downloaded
    new_folder: str
        name of the folder where the data is to be saved
    dl_stdmet: bool
        download standard meteorological data
    dl_cwind: bool
        download continuous winds data
    dl_swden: bool 
        download spectral wave density data
    dl_swdir: bool
        download spectral wave (alpha1) direction data
    dl_swdir2: bool
        download spectral wave (alpha2) direction data
    dl_swr1: bool
        download spectral wave (r1) direction data
    dl_swr2: bool
        download spectral wave (r2) direction data
    
    Returns:
    -----------
    None
    """
    urls = []

    for year in years:
        # Standard meteorological data 
        if dl_stdmet:
            type = "stdmet"
            letter = "h"
            # creating a url of the following form:
            # 'https://www.ndbc.noaa.gov/view_text_file.php?filename=44008h2015.txt.gz&dir=data/historical/stdmet/'
            start = 'https://www.ndbc.noaa.gov/view_text_file.php?filename='
            middle = '.txt.gz&dir=data/historical/'
            url =  start + str(buoy_id) + letter + str(year) + middle + type + '/'

            urls.append(url)
        
        # Continuous winds data
        if dl_cwind:
            type = "cwind"
            letter = "c"
            # creating a url of the following form:
            # 'https://www.ndbc.noaa.gov/view_text_file.php?filename=44008h2015.txt.gz&dir=data/historical/stdmet/'
            start = 'https://www.ndbc.noaa.gov/view_text_file.php?filename='
            middle = '.txt.gz&dir=data/historical/'
            url =  start + str(buoy_id) + letter + str(year) + middle + type + '/'

            urls.append(url)

        # Spectral wave density data
        if dl_swden:
            type = "swden"
            letter = "w"
            # creating a url of the following form:
            # 'https://www.ndbc.noaa.gov/view_text_file.php?filename=44008h2015.txt.gz&dir=data/historical/stdmet/'
            start = 'https://www.ndbc.noaa.gov/view_text_file.php?filename='
            middle = '.txt.gz&dir=data/historical/'
            url =  start + str(buoy_id) + letter + str(year) + middle + type + '/'

            urls.append(url)

        # Spectral wave (alpha1) direction data
        if dl_swdir:
            type = "swdir"
            letter = "d"
            # creating a url of the following form:
            # 'https://www.ndbc.noaa.gov/view_text_file.php?filename=44008h2015.txt.gz&dir=data/historical/stdmet/'
            start = 'https://www.ndbc.noaa.gov/view_text_file.php?filename='
            middle = '.txt.gz&dir=data/historical/'
            url =  start + str(buoy_id) + letter + str(year) + middle + type + '/'

            urls.append(url)

        # Spectral wave (alpha2) direction data
        if dl_swdir2:
            type = "swdir2"
            letter = "i"
            # creating a url of the following form:
            # 'https://www.ndbc.noaa.gov/view_text_file.php?filename=44008h2015.txt.gz&dir=data/historical/stdmet/'
            start = 'https://www.ndbc.noaa.gov/view_text_file.php?filename='
            middle = '.txt.gz&dir=data/historical/'
            url =  start + str(buoy_id) + letter + str(year) + middle + type + '/'

            urls.append(url)

        # Spectral wave (r1) direction data
        if dl_swr1:
            type = "swr1"
            letter = "j"
            # creating a url of the following form:
            # 'https://www.ndbc.noaa.gov/view_text_file.php?filename=44008h2015.txt.gz&dir=data/historical/stdmet/'
            start = 'https://www.ndbc.noaa.gov/view_text_file.php?filename='
            middle = '.txt.gz&dir=data/historical/'
            url =  start + str(buoy_id) + letter + str(year) + middle + type + '/'

            urls.append(url)

        # Spectral wave (r2) direction data
        if dl_swr2:
            type = "swr2"
            letter = "k"
            # creating a url of the following form:
            # 'https://www.ndbc.noaa.gov/view_text_file.php?filename=44008h2015.txt.gz&dir=data/historical/stdmet/'
            start = 'https://www.ndbc.noaa.gov/view_text_file.php?filename='
            middle = '.txt.gz&dir=data/historical/'
            url =  start + str(buoy_id) + letter + str(year) + middle + type + '/'

            urls.append(url)

        if dl_cwind:
            type = "cwind"
            letter = "c"
            # creating a url of the following form:
            # 'https://www.ndbc.noaa.gov/view_text_file.php?filename=44008h2015.txt.gz&dir=data/historical/stdmet/'
            start = 'https://www.ndbc.noaa.gov/view_text_file.php?filename='
            middle = '.txt.gz&dir=data/historical/'
            url =  start + str(buoy_id) + letter + str(year) + middle + type + '/'

            urls.append(url)

    # Make a new folder for the buoy data if it does not exist
    full_path = os.path.join(folder_path,new_folder)
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    # Download data
    for url in urls:
        equal_idx = url.find("=")
        txt_idx = url.find(".txt", equal_idx)
        type = url.split('/')[-2]
        file_name = url[equal_idx+1:txt_idx] + "_" + type + ".txt"
        output_file = os.path.join(full_path,file_name)

        # check if the file already exists
        if os.path.exists(output_file):
            print(f"File {file_name} already exists. Skipping download.")
            continue

        retry_count = 3
        retry_delay = 5  # seconds

        for _ in range(retry_count):
            print(f"Retrieving data for buoy {buoy_id} from {url}")

            response = requests.get(url)

            if response.status_code == 200:
                with open(output_file, 'w', encoding='utf-8') as file:
                    file.write(response.text)
                
                print(f"{file_name} data saved to {output_file}")
                break  # Break out of the retry loop if successful
            else:
                print(f"Failed to retrieve the webpage for file {file_name}. Status code: {response.status_code}")

            # Adding a delay before retrying
            time.sleep(retry_delay)
        else:
            print("Maximum retries exceeded. Failed to retrieve data.")
        

    return None

def convert_NDBC_to_netcdf(folder_path,buoy_id,overwrite=False):
    """
    Convert NDBC text files to one netCDF file using the first row as the new variable names, 
    skipping the second row (units), and combining the first 5 columns into a datetime64 object.

    KK MIT/WHOI 2024

    Arguments:
    -----------
    folder_path: str
        folder where the text files are located
    overwrite: bool
        overwrite existing files
    
    Returns:
    -----------
    output_file: str
        path to the downloaded file
    """
    
    # Exit the function if the file already exists and overwrite is False
    if os.path.exists(os.path.join(folder_path, f'{buoy_id}.nc')) and not overwrite:
        print(f"File {buoy_id}.nc already exists. Skipping conversion.")
        return None

    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Create an empty list to store the data
    data = []

    # Loop through the files and read the data
    for file in files:
        # Skip files that are not text files
        if not file.endswith('.txt'):
            continue

        # Read the data from the file
        with open(os.path.join(folder_path, file), 'r') as f:
            lines = f.readlines()

        # Get the variable names from the first row
        var_names = lines[0].strip().split()

        # Get the data from the remaining rows
        for line in lines[2:]:
            data.append(line.strip().split())

    # Convert missing values to NaN
    data = np.array(data)
    data[data == '9999.0'] = np.nan
    data[data == '999.0'] = np.nan
    data[data == '99.00'] = np.nan
    data[data == '99.0'] = np.nan

    # Convert the data to a numpy array
    data = np.array(data)

    # Create a new NetCDF file
    output_file = os.path.join(folder_path,f'{buoy_id}stdmet.nc')
    print(f"Converting data to NetCDF format and saving to {output_file}")
    nc = nc4.Dataset(output_file, 'w')

    # Create dimensions
    num_records = data.shape[0]
    num_variables = data.shape[1] - 5  # Exclude first 5 columns for time
    nc.createDimension('time', num_records)
    nc.createDimension('variables', num_variables)

    # Create variables
    time_var = nc.createVariable('time', 'i4', ('time',))
    time_var.units = 'seconds since 1970-01-01 00:00:00'

    # Populate time variable
    time_data = []
    for row in data:
        time_str = ' '.join(row[:5])
        dt_obj = datetime.strptime(time_str, '%Y %m %d %H %M')
        timestamp = int(dt_obj.timestamp())
        time_data.append(timestamp)
    time_var[:] = np.array(time_data, dtype=np.int32)

    # Create other variables
    for i, var_name in enumerate(var_names[5:]):
        var_data = data[:, i + 5].astype(float)
        var = nc.createVariable(var_name, 'f4', ('time',))
        var[:] = var_data

    # Add global attributes
    nc.title = 'NDBC Buoy Data'
    nc.source = 'National Data Buoy Center (NDBC)'
    nc.history = 'Converted to NetCDF using custom Python script'

    # Close the NetCDF file
    nc.close()

    print(f"Data saved to {output_file}")

    return output_file
    

def download_CDIP_data(buoy_id, folder_path, new_folder, archive, realtime, overwrite=False):
    """
    Download data from the Coastal Data Information Program (CDIP) website for a given station and years

    KK MIT/WHOI 2024

    Arguments:
    -----------
    buoy_id: int
        ID of the station to download data from
    folder_path: str
        folder where the data is to be downloaded
    new_folder: str
        name of the folder where the data is to be saved
    Archived: bool
        download archived data
    Realtime: bool
        download realtime data
    overwrite: bool
        overwrite existing files
    
    Returns:
    -----------
    output_file: str
        path to the downloaded file
    """
   
    # Create the links to the data
    if archive:
        # data_url = 'https://thredds.cdip.ucsd.edu/thredds/fileServer/cdip/archive/' + buoy_id + 'p1/' + buoy_id + 'p1_historic.nc'
        data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/' + buoy_id + 'p1/' + buoy_id + 'p1_historic.nc'
        output_file = os.path.join(folder_path,new_folder,buoy_id + 'p1_historic.nc')
    if realtime:
        data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/' + buoy_id + 'p1_rt.nc'

    # Make a new folder for the buoy data if it does not exist
    full_path = os.path.join(folder_path,new_folder)
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    # Check to see if the file already exists and overwrite is False
    if os.path.exists(output_file) and not overwrite:
        print(f"File {output_file} already exists. Skipping download.")
        return None    
    
    # Open the data file as a netCDF dataset
    nc = nc4.Dataset(data_url, 'r')
    
    # Create a new NetCDF file for writing
    output_nc = nc4.Dataset(output_file, 'w')

    # Copy dimensions from the input NetCDF file to the output file
    for dim_name, dim in nc.dimensions.items():
        output_nc.createDimension(dim_name, len(dim) if not dim.isunlimited() else None)

    # Copy variables and their attributes
    for var_name, var in nc.variables.items():
        output_var = output_nc.createVariable(var_name, var.dtype, var.dimensions, fill_value=var._FillValue if '_FillValue' in var.ncattrs() else None)
        output_var[:] = var[:]  # Copy variable data
        # Copy variable attributes
        for attr_name in var.ncattrs():
            if attr_name != '_FillValue':  # Skip _FillValue attribute
                output_var.setncattr(attr_name, var.getncattr(attr_name))

    # Close both NetCDF datasets
    nc.close()
    output_nc.close()

    # # Open the data file as a netCDF dataset
    # nc = xr.open_dataset(data_url)
    # # print('datakeys:', nc.variables.keys())

    # # Save the data to a .nc file
    # nc.to_netcdf(output_file)

    # # Close the netCDF dataset
    # nc.close()

    print(f"File downloaded and saved to {output_file}")

    return output_file


def download_WIS_data(buoy_id, folder_path, new_folder, WIS_region, WIS_subregion,overwrite=True):
    """
    Download data from the Wave Information Study (WIS) website for a given buoy

    KK MIT/WHOI 2024

    Arguments:
    -----------
    buoy_id: int
        ID of the buoy to download data from
    folder_path: str
        folder where the data is to be downloaded
    new_folder: str
        name of the folder where the data is to be saved
    WIS_region: str
        region of the data
    WIS_subregion: str
        subregion of the data
    overwrite: bool
        overwrite existing files
    
    Returns:
    -----------
    output_file: str
        path to the downloaded file
    """

    # Check if the subregion is Japan
    if WIS_subregion == 'Japan':
        print('Japan files not yet implemented')
        return None
    # Create the link to the data
    else:
        data_url = 'https://chldata.erdc.dren.mil/thredds/dodsC/wis/' + WIS_region + '/' + str(buoy_id) + '/' + str(buoy_id) + '.nc4'
        output_file = os.path.join(folder_path,new_folder,buoy_id + '.nc4')

    # Make a new folder for the buoy data if it does not exist
    full_path = os.path.join(folder_path,new_folder)
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    # Check to see if the file already exists and overwrite is False
    if os.path.exists(output_file) and not overwrite:
        print(f"File {output_file} already exists. Skipping download.")
        return None

    # Open the data file as a netCDF dataset
    nc = nc4.Dataset(data_url, 'r')
    
    # Create a new NetCDF file for writing
    output_nc = nc4.Dataset(output_file, 'w')

    # Copy dimensions from the input NetCDF file to the output file
    for dim_name, dim in nc.dimensions.items():
        output_nc.createDimension(dim_name, len(dim) if not dim.isunlimited() else None)

    # Copy variables and their attributes
    for var_name, var in nc.variables.items():
        output_var = output_nc.createVariable(var_name, var.dtype, var.dimensions, fill_value=var._FillValue if '_FillValue' in var.ncattrs() else None)
        output_var[:] = var[:]  # Copy variable data
        # Copy variable attributes
        for attr_name in var.ncattrs():
            if attr_name != '_FillValue':  # Skip _FillValue attribute
                output_var.setncattr(attr_name, var.getncattr(attr_name))

    # Close both NetCDF datasets
    nc.close()
    output_nc.close()

    print(f"File downloaded and saved to {output_file}")

    return output_file


def get_file_size(buoy_id,org,region,subregion,target_row_index,target_column_index):
    """
    Get the size of a file from the the thredds catalog

    KK MIT/WHOI 2024

    Arguments:
    -----------
    buoy_id: int
        ID of the buoy to download data from
    org: str
        organization that hosts the data
    region: str
        region of the data (WIS ONLY)
    target_row_index: int
        row index of the file size in the catalog
    target_column_index: int
        column index of the file size in the catalog

    Returns:
    -----------
    file_size: str
        size of the file
    """
    if org == 'CDIP':
        catalog_link = 'http://thredds.cdip.ucsd.edu/thredds/catalog/cdip/archive/' + str(buoy_id) + 'p1/catalog.html'
        catalog = open_url(catalog_link)
        rows = catalog.find_all('tr')
        file_size = rows[target_row_index].find_all('td')[target_column_index].get_text().strip()

    if org == 'WIS':
        if subregion == 'Japan':
            print('Japan files not yet implemented')
            return None
        else:
            catalog_link = 'https://chldata.erdc.dren.mil/thredds/catalog/wis/' + region + '/' + str(buoy_id) + '/catalog.html'
            catalog = open_url(catalog_link)
            rows = catalog.find_all('tr')
            file_size = rows[target_row_index].find_all('td')[target_column_index].get_text().strip()
    return file_size
