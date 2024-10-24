from typing import Dict, List
from math import *
import re
import polyline
import pandas as pd
import numpy as np


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    main=[]
    for k in range(0,len(ip_lst),n):
        tmp_l = []
        for l in range(k,min(k+n,len(ip_lst))):
            tmp_l.insert(0,ip_lst[l])
        main.extend(tmp_l)
    return main


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    main_dct = {}
    for i in ip_lst:
        if len(i) not in main_dct.keys():
            main_dct.update({len(i):[i]})
        else:
            main_dct[len(i)].append(i)
    return dict(sorted(main_dct.items()))


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    tmp_l = []
    for a, b in ip_dictionary.items():
        key = f"{main_key}{sep}{a}" if main_key else a
        if isinstance(b, dict):
            tmp_l.extend(flatten(b, key, sep=sep).items())
        elif isinstance(b, list):
            for i, item in enumerate(b):
                tmp_l.extend(flatten({f'{a}[{i}]': item}, main_key, sep=sep).items())
        else:
            tmp_l.append((key, b))
    return dict(tmp_l)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(base_path, ele):
        if len(base_path) == len(input_l):
            res.append(base_path[:])
            return
        for i in range(len(input_l)):
            if ele[i]:
                continue
            if i > 0 and input_l[i] == input_l[i - 1] and not ele[i - 1]:
                continue
            ele[i] = True
            base_path.append(input_l[i])
            backtrack(base_path, ele)
            base_path.pop()
            ele[i] = False

    input_l.sort()  
    res = []
    ele = [False] * len(input_l)
    backtrack([], ele)
    return res


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_patterns = [
        r"\b(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-(\d{4})\b",     # dd-mm-yyyy
        r"\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(\d{4})\b",     # mm/dd/yyyy
        r"\b(\d{4})\.(0[1-9]|1[0-2])\.(0[1-9]|[12][0-9]|3[01])\b"    # yyyy.mm.dd
    ]
    
    combined_pattern = "|".join(date_patterns)
    
    
    matches = re.findall(combined_pattern, input_string)
    
    
    valid_dates = []
    for match in matches:
        
        if match[0] and match[1] and match[2]:  
            valid_dates.append(f"{match[0]}-{match[1]}-{match[2]}")
        elif match[3] and match[4] and match[5]:  
            valid_dates.append(f"{match[3]}/{match[4]}/{match[5]}")
        elif match[6] and match[7] and match[8]:  
            valid_dates.append(f"{match[6]}.{match[7]}.{match[8]}")
    
    return valid_dates

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    def haversine_formula(lat1, lon1, lat2, lon2):
        R = 6371000  
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # print(lat1, lon1, lat2, lon2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # print(dlat,dlon)
        
        a = sin(dlat / 2)*2 + cos(lat1) * cos(lat2) * sin(dlon / 2)*2
        # print(a)
        # print(sqrt(a),sqrt(1 - a))
        a = 0.00 if a < 0 else abs(a)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        return R * c

    # Decode polyline into coordinates (latitude, longitude)
    # print(polyline_str)
    coordinates = polyline.decode(polyline_str)
    # print(coordinates)
    
    # Create DataFrame
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Initialize distance column
    df['distance'] = 0.0
    
    # Calculate distances between successive coordinates
    for i in range(1, len(df)):
        df.loc[i, 'distance'] = haversine_formula(df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude'],
                                          df.loc[i, 'latitude'], df.loc[i, 'longitude'])
        
    
    return df



def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    def rotate_matrix_90_clockwise(matrix):
        n = len(matrix)
        rotated_matrix = [[0] * n for _ in range(n)]
        
        # Rotate the matrix by 90 degrees clockwise
        for i in range(n):
            for j in range(n):
                rotated_matrix[j][n - 1 - i] = matrix[i][j]
        
        return rotated_matrix

    def sum_of_row_and_col_excluding_element(matrix, row, col):
        n = len(matrix)
        row_sum = sum(matrix[row]) - matrix[row][col]
        col_sum = sum(matrix[i][col] for i in range(n)) - matrix[row][col]
        return row_sum + col_sum

    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = rotate_matrix_90_clockwise(matrix)
    
    # Step 2: Replace each element with the sum of all elements in the same row and column, excluding itself
    transformed_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            transformed_matrix[i][j] = sum_of_row_and_col_excluding_element(rotated_matrix, i, j)
    
    return transformed_matrix


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    weekday_No  = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }

    # Convert the weekdays to numbers
    df['start_weekday'] = df['startDay'].map(weekday_No)
    df['end_weekday'] = df['endDay'].map(weekday_No)

    # Create a reference date for the current week
    Today = datetime.now()
    start_of_week = Today - timedelta(days=Today.weekday())  # Get last Monday

    # Create actual dates for start and end days based on weekdays
    df['start_date'] = df['start_weekday'].apply(lambda x: start_of_week + timedelta(days=x))
    df['end_date'] = df['end_weekday'].apply(lambda x: start_of_week + timedelta(days=x))

    # Combine dates and times into datetime objects
    df['start_datetime'] = pd.to_datetime(df['start_date'].dt.strftime('%Y-%m-%d') + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['end_date'].dt.strftime('%Y-%m-%d') + ' ' + df['endTime'])

    # Create a multi-index DataFrame grouped by (id, id_2)
    grouped = df.groupby(['id', 'id_2'])

    def check_group(grp):
        # Collect unique weekdays
        unique_start_days = set(grp['start_weekday'])
        unique_end_days = set(grp['end_weekday'])
        Union_days = unique_start_days.union(unique_end_days)

        # Check if all 7 weekdays are covered
        all_weekdays = {0, 1, 2, 3, 4, 5, 6}  # Monday=0, ..., Sunday=6
        Num_of_days = len(Union_days) == 7

        # Get the earliest start and latest end
        start_range = grp['start_datetime'].min()
        end_range = grp['end_datetime'].max()
        
        # Check if the timestamps cover a full 24-hour period
        full_day_covered = (end_range - start_range) >= timedelta(days=1)

        return not ( Num_of_days and full_day_covered)

    # Apply the check_group function to each group and return as a boolean series
    result = grouped.apply(check_group)

    return result
