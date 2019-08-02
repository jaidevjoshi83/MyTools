import glob
import pandas as pd 
import sys

files = sys.argv[1]
out_file = sys.argv[2]
class_label = sys.argv[3]

Update_df = pd.read_csv(files.split(',')[0],sep='\t')

for file in files.split(',')[1:]: 

    second_df = pd.read_csv(file,sep='\t')
    Update_df =  pd.concat([Update_df,second_df])

Update_df = Update_df.fillna(0)

print (Update_df.shape)
cl = Update_df.columns.tolist()
Update_df = Update_df.values
Update_df = pd.DataFrame(Update_df,columns=cl)


if class_label == "None":

    final_df = Update_df
    final_df.to_csv(out_file,sep="\t", index=False)

else:

    lable_df = pd.DataFrame([class_label]*Update_df.shape[0],columns=['class_label'])
    final_df = pd.concat([Update_df,lable_df],axis=1)
    final_df.to_csv(out_file,sep="\t", index=False)