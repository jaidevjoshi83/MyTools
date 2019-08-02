import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import os,sys
"""
We will be using Extra Tree Classifier for extracting the top 10 features for the dataset.
"""
def Tree_based_Feature_selection(in_data,Out_data_frame,report_file,out_html_dir,No_of_features):


    data = pd.read_csv(in_data,sep='\t')
    Col_list = data.columns.tolist()

    if int(No_of_features) > len(Col_list):
        print("args value 'No_of_features' is larger then the total feature present in Data File") 
        sys.exit(0)
    else:
        if not os.path.exists(os.path.join(os.getcwd(), out_html_dir)):
            os.makedirs(os.path.join(os.getcwd(), out_html_dir))
        pass

    X = data[Col_list[0:len(Col_list)-1]]
    Y_train = data[Col_list[len(Col_list)-1]]
    scaler = preprocessing.StandardScaler().fit(X)
    X_train = scaler.transform(X)

    model = ExtraTreesClassifier()
    model.fit(X_train,Y_train)
    print(model.feature_importances_) 

    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(int(No_of_features)).plot(kind='barh')
    feat_importances =  feat_importances.nlargest(int(No_of_features)).to_frame()
    feat_importances.index.values
    reduced_data_Frame = pd.concat([data[feat_importances.index.values],Y_train],axis=1)
    reduced_data_Frame.to_csv(Out_data_frame, index=None,sep='\t')

    plt.savefig(os.path.join(os.getcwd(), out_html_dir,'Top_features.png'), format='png')
    #plt.show()


    out_html = open(os.path.join(out_html_dir, report_file),'w')              

    part_1 ="""

    <!DOCTYPE html>
    <html lang="en">
    <head>
      <title>Bootstrap Example</title>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css">
      <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.0/jquery.min.js"></script>
      <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>
    
    <body>

    <style>
    div.container_1 {
      width:600px;
      margin: auto;
     padding-right: 10; 

    }

    </style>
   </head>
    <div class="jumbotron text-center">
      <h1> Tree Based Feature Selection </h1>
    </div>
      

    
    <div class="container">

      

      <div class="row">

        <div class="col-sm-4">

        <h2>Top Features</h2>
         
      
          <img src="Top_features.png" alt="Smiley face" height="600" width="800">
       
        
        </div>

        
      </div>
    </div>

    </body>
    </html>
    """ 



    out_html.write(part_1)
    out_html.close()




if __name__=="__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument("-I", "--Input_data",
                            required=True,
                            default=None,
                            help="Path to target CSV file")
                            
    parser.add_argument("-O", "--output_data_file",
                        required=True,
                        default=None,
                        help="Path to out file")
    
    parser.add_argument("-R", "--out_html_repot",
                        required=True,
                        default='report_file.html',
                        help="Path to out file")

    parser.add_argument("-D", "--out_report_dir_name",
                required=None,
                default=os.path.join(os.getcwd(), 'report_dir'),   
                help="Path to out file")

    parser.add_argument("-N", "--No_of_top_features",
                required=None,
                default=20,   
                help="Path to out file")


    args = parser.parse_args()
    Tree_based_Feature_selection(args.Input_data, args.output_data_file, args.out_html_repot, args.out_report_dir_name,args.No_of_top_features)


