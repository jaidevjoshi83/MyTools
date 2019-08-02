print ("""

    ################################################################
    #        Python Script to select feature for ML modeling       #
    ################################################################
    ###       Modified From the PEP_learn_project                ###
    ################################################################
    #  USES:                                                       #
    #  python Feature_selection.py -t test.csv -c 0.6 --o out.csv  #
    ################################################################
    """)

import numpy as np
#import pylab as pl
import pandas as pd
from sklearn import datasets, svm
from sklearn.feature_selection import SelectFpr, f_classif
import os
import matplotlib as plt
import matplotlib.pylab as pl


class feature(object):
    
         
    def data_gen(self, in_file):
        
        self.in_file = in_file
        self.df = pd.read_csv(self.in_file,sep='\t')
        #self.df = self.df.sample('100') 
        self.clm_list = []
        
        for column in self.df.columns:
            self.clm_list.append(column)

        x = self.df[self.clm_list[0:len(self.clm_list)-1]].values
        y = self.df[self.clm_list[len(self.clm_list)-1]].values
        x_indices = np.arange(x.shape[-1])
        
        return x, y, x_indices, self.clm_list, self.df
        

    def svm_feature_Selection(self, x_Data, y_Data):

        clf = svm.SVC(kernel='linear')

        print (x_Data)
        print (y_Data)
        clf.fit(x_Data, y_Data)
        svm_weights = (clf.coef_**2).sum(axis=0)
        svm_weights /= svm_weights.max()

        return  svm_weights

    def plot2(self, xval,o_dir):

        conter = 0
        ylist = []
        for x in xval:
            conter = conter+1
            ylist.append(x)
        xlist  = range(0, conter)
        ax = pl.subplot(111)
        ax.bar(xlist, ylist, width=5, label='SVM weight', color='b')
        pl.axis('tight')
        pl.xlabel('OTU number')
        pl.ylabel('Weight')
        pl.legend(loc='upper right')
        pl.savefig(os.path.join(o_dir,'Selected_feature.png'))
        pl.close()

    def feature_selection_main(self, infile, result_sum, cut_off,output_dir,report):
        
        self.infile = infile
        self.result_sum = result_sum
        self.cut_off =  cut_off
        self.selected_out = []
        self.des_list = []
        self.data_x, self.data_y, self.x_ind, self.clm_list, self.df = feature().data_gen(self.infile)
        self.output_dir = output_dir
        self.report = report

        if not os.path.exists(os.path.join(os.getcwd(), self.output_dir)):
            os.makedirs(os.path.join(os.getcwd(), self.output_dir))


        svm_wieghts = feature().svm_feature_Selection(self.data_x, self.data_y)
        sno = 0

        #for n in svm_wieghts:
           # print(n)  
        
        for i, svmw in enumerate(svm_wieghts):
            if svmw >= float(self.cut_off):

                print(svmw)
                sno = sno + 1

                self.selected_out.append([sno, self.x_ind[i+1], round(svmw, 1), self.clm_list[i]])
                self.des_list.append(self.clm_list[i])
       
        self.predf = pd.DataFrame(self.selected_out, columns=["Sn","feature index", "svm_wieght", "feature_name"])
        #self.predf.to_csv(self.result_sum,index = False)


        print ("Result written to the file ! ---> ", self.result_sum)
        self.df.to_csv(self.result_sum, columns = self.des_list+[self.clm_list[len(self.clm_list)-1]], index = False,sep='\t')
        print ("Selected features written to the file ! ---> ", "reduced.data.csv")
        feature().plot2(svm_wieghts,output_dir)

        out_html = open(os.path.join(self.output_dir, self.report),'w')              

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
          <h1> Univariate Feature Selection </h1>
        </div>
          

        
        <div class="container">

          

          <div class="row">

            <div class="col-sm-4">

            <h2>Top Features</h2>
             
          
              <img src="Selected_feature.png" alt="Smiley face" height="600" width="800">
           
            
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
    
    parser.add_argument("-t", "--train",
                        required=True,
                        default=None,
                        help="Path to training data as csv file")
                        
    parser.add_argument("-c", "--cutoff",
                        required=False,
                        default=0.1,
                        help="cutoff value for feature selection (between 0 to 1, default is 0.5)")   

    parser.add_argument("-r", "--report",
                        required=True,
                        default=None,
                        help="html report file")  

                        
    parser.add_argument("-o", "--out",
                        required=False,
                        default = "selected_feature.csv",
                        help="output file name") 

    parser.add_argument("-d", "--out_dir_name",
                required=None,
                default=os.path.join(os.getcwd(), 'report_dirr'),   
                help="Path to out file")
                        
    args = parser.parse_args()
    c = feature()
    c.feature_selection_main(args.train, args.out, args.cutoff, args.out_dir_name, args.report)