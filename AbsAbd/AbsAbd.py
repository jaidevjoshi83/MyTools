import pandas as pd 
import os


def Abs_abd(in_files,out_file_dir):


    if not os.path.exists(out_file_dir):
            os.makedirs(out_file_dir)


    for i,in_file in enumerate(in_files.split(',')):




        f = open(in_file)
        lines = f.readlines()

        column_name = []
        values = []
        final_values = []

        for line in lines[2:]:
            column_name.append(line.split('\t')[0]),values.append(float(line.split('\t')[1]))

        for value in values:
            final_values.append(round(value/sum(values),5))

        df = pd.DataFrame([final_values],columns=column_name)

        out_file = os.path.join(out_file_dir,'%i_out.tsv'%i)
        df.to_csv(out_file,sep='\t',index=None)


if __name__=="__main__":


    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-I", "--in_file",
                        required=True,
                        default=None,
                        help="Path to target TSV file")
                        
    parser.add_argument("-O", "--out_file_dir",
                        required=True,
                        default='out_file_dir',
                        help="Path to target tsv file")
                       
    args = parser.parse_args()


    Abs_abd(args.in_file,args.out_file_dir)

