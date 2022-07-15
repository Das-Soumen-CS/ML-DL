import csv 
import json 
import sys
import pandas as pd 
import os
import pathlib

def CSV_2_JSON(file_path_CSV ,file_path_JSON):
    
    #creating Data Dictionary

    Data_Dictionary_JSON={}

    #open a CSV file  handler

    with  open (file_path_CSV ,encoding ='utf-8') as csv_file :
         csv_reader=csv.DictReader(csv_file)

         #convert each row into dictionary and add it to "Data_Dictionary_JSON"

         for rows in csv_reader:
            #assuming a column named 'id'to be the primary key
            #key = rows['id']
            key = rows['question']
            Data_Dictionary_JSON[key] = rows
            
    #open a json file handler and use json.dumps
    #method to dump the data
    #Step 3

    with open (file_path_JSON ,'w',encoding ='utf-8') as json_file:
        json_file.write(json.dumps(Data_Dictionary_JSON,indent =4))
    print(json.dumps(Data_Dictionary_JSON,indent =4))


def main():

    # From where need to read the CSV file
    file_path_CSV = sys.argv[1]
    # where to store the JSON file
    file_path_JSON=os.path.dirname(file_path_CSV)
    #print(file_path_JSON)
    prefix=os.path.basename(file_path_CSV)
    prefix=os.path.splitext(prefix)[0]
    #print(prefix)
    #file_path_JSON = os.rename(file_path_CSV, destination + '.json')
    file_path_JSON = os.path.join( file_path_JSON , prefix+".json")
    print(file_path_JSON)

    df =pd.read_csv(file_path_CSV)
    print(df)
    # call CSV_2_JSON
    CSV_2_JSON(file_path_CSV ,file_path_JSON)


main()

