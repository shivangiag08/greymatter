import numpy as np
import pandas as pd
# PTID = ['003_S_6014', '003_S_6644', '003_S_1122', '003_S_5154', '003_S_4081', '003_S_6258', '003_S_6256', '002_S_1280', '003_S_6268', '003_S_6259', '003_S_4119', '005_S_4168', '003_S_4644', '003_S_4441', '003_S_4872', '003_S_4288', '002_S_6007', '003_S_4900', '003_S_0908', '003_S_6067']
PTID=data = [
    "098_S_4018", "098_S_4017", "116_S_1271", "031_S_0294", "031_S_4021", 
    "023_S_4020", "031_S_4024", "099_S_4022", "116_S_4010", "037_S_4028",
    "024_S_4084", "067_S_4782", "011_S_4827", "014_S_2185", "014_S_4401", 
    "022_S_6069", "041_S_4060", "041_S_4138", "041_S_4143", "041_S_4874",
    "011_S_0002", "011_S_0003", "011_S_0005", "011_S_0008", "022_S_0007", 
    "100_S_0015", "023_S_0030", "023_S_0031", "011_S_0016", "011_S_0021"
]
ADAS_scores = pd.read_csv('/Users/Agaaz/Downloads/ADAS_scores.csv')
ADAS_scores = ADAS_scores[ADAS_scores.PTID.isin(PTID)]
ADAS_scores.to_csv('/Users/Agaaz/Downloads/filtered_ADAS_scores1.csv', index=False)

MMSE_scores = pd.read_csv('/Users/Agaaz/Downloads/MMSE_scores.csv')
MMSE_scores = MMSE_scores[MMSE_scores.PTID.isin(PTID)]
MMSE_scores.to_csv('/Users/Agaaz/Downloads/filtered_MMSE_scores1.csv', index=False)