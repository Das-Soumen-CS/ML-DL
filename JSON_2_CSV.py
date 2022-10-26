import csv 
import json 
import sys 
import pandas as pd
import numpy as np 
import os 
import matplotlib.pyplot as plt 


def JSON_2_CSV(json_data,file_path_csv):
    json_data=json_data.to_json(orient="index")
    parsed = json.loads(json_data)
    print("\n JSON File before Conversion =  \n" ,json.dumps(parsed,indent =4))
    # To Convert From JSON to CSV and and store with same name in same directory eg: data_1.json to data_1.csv
    pd_Obj = pd.read_json(json_data, orient='index')
    csvData = pd_Obj.to_csv(file_path_csv,index=False)
    print("\n CSV file after Conversion =\n\n",pd.read_csv(file_path_csv))
    pass

def main():
    file_path_json=sys.argv[1]
    json_data=pd.read_json(file_path_json,orient='index')
    # To strore json_2_csv file in same directory creating "file_path_csv"
    file_path_csv=os.path.dirname(file_path_json)
    prefix=os.path.basename(file_path_json)
    prefix=os.path.splitext(prefix)[0]
    file_path_csv = os.path.join( file_path_csv, prefix+".csv")
    #print(file_path_csv)  
    JSON_2_CSV(json_data,file_path_csv)
    pass

main()