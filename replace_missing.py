import pandas as pd 
import numpy as np

def replace_missing2(col1, col2):
    if pd.isna(col1):
        return col2
    else: 
        return col1

def replace_missing4(col1, col2, col3, col4):
    if pd.isna(col1) & pd.isna(col2) & pd.isna(col3):
        return col4
    else: 
        if pd.isna(col1) & pd.isna(col2):
            return col3
        else:
            if pd.isna(col1):
                return col2
            else: 
                return col1

def replace_missing (col1, col2, col3, col4, col5):
    if pd.isna(col1) & pd.isna(col2) & pd.isna(col3) & pd.isna(col4):
        return col5
    else:
        if pd.isna(col1) & pd.isna(col2) & pd.isna(col3):
            return col4
        else: 
            if pd.isna(col1) & pd.isna(col2):
                return col3
            else:
                if pd.isna(col1):
                    return col2
                else: 
                    return col1

def replace_missing6 (col1, col2, col3, col4, col5, col6):
    if pd.isna(col1) & pd.isna(col2) & pd.isna(col3) & pd.isna(col4) & pd.isna(col5):
        return col6
    else:
        if pd.isna(col1) & pd.isna(col2) & pd.isna(col3) & pd.isna(col4):
            return col5
        else:
            if pd.isna(col1) & pd.isna(col2) & pd.isna(col3):
                return col4
            else: 
                if pd.isna(col1) & pd.isna(col2):
                    return col3
                else:
                    if pd.isna(col1):
                        return col2
                    else: 
                        return col1


def replace_missing7 (col1, col2, col3, col4, col5, col6, col7):
    if pd.isna(col1) & pd.isna(col2) & pd.isna(col3) & pd.isna(col4) & pd.isna(col5) & pd.isna(col6):
        return col7
    else: 
        if pd.isna(col1) & pd.isna(col2) & pd.isna(col3) & pd.isna(col4) & pd.isna(col5):
            return col6
        else:
            if pd.isna(col1) & pd.isna(col2) & pd.isna(col3) & pd.isna(col4):
                return col5
            else:
                if pd.isna(col1) & pd.isna(col2) & pd.isna(col3):
                    return col4
                else: 
                    if pd.isna(col1) & pd.isna(col2):
                        return col3
                    else:
                        if pd.isna(col1):
                            return col2
                        else: 
                            return col1