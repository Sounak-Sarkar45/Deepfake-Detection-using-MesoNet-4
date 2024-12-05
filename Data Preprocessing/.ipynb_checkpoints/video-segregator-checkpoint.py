import pandas as pd
import shutil
import os
df = pd.read_json("C:/Users/LENOVO/Downloads/dfdc_train_part_0/metadata.json").T
print(f'The shape of the data frame: {df.shape}')
# df.to_csv('name.csv')
source = 'C:/Users/LENOVO/Downloads/dfdc_train_part_0'
destination = 'D:/WORK/UNIVERSITY PROJECTS/SET-3/VIDEO'
for index, row in df.iterrows():
    source_path = os.path.join(source,index)
    class_folder = os.path.join(destination, row['label'])
    destination_path = os.path.join(class_folder,index)
    try:
        shutil.move(source_path,destination_path)
    except:
        print(index,row)
        print('='*50)