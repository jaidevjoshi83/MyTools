import numpy as np
import sys,os
from scipy import interp
import pylab as pl
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import *
import matplotlib
############Changes###############################
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#################################################
import matplotlib.image as mpimg



class My_csv_class(object):

    def data_gen(self,csv_path):
        
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path,sep='\t')

        clm_list = []
        for column in self.df.columns:
            clm_list.append(column)

        X_data = self.df[clm_list[0:len(clm_list)-1]].values
        y_data = self.df[clm_list[len(clm_list)-1]].values

        return X_data.astype(float), y_data
        
    def roc_gen(self, X, y, model,nfolds,st,dir_path,image_out_path,Selected_Sclaer):
        
        self.st = st
        self.nfolds = nfolds
        self.dir_path = dir_path
        self.image_out_path = image_out_path
        self.Selected_Sclaer = Selected_Sclaer

        
        if self.nfolds > 10:
            print ("nfolds is too high")
            sys.exit()
            
        elif self.nfolds <= 2:
            print ("nfolds value is too small ")
            sys.exit()

        elif self.nfolds > 5:
            
             print("################################### Warrning ##########################################") 
             print(" Please make sure that your data is large Enough to Properly Handle the high nfolds,>5 ")
             print("#######################################################################################")
               
        else:
            pass

        specificity_list = []
        sensitivity_list = []
        presison_list = []
        mcc_list =  []
        f1_list = []

        folds = StratifiedKFold(n_splits=5)
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        for i, (train, test) in enumerate(folds.split(X, y)):

            ############Changes###############################

            if Selected_Sclaer=='Min_Max':
                scaler = MinMaxScaler().fit(X[train])
                x_train = scaler.transform(X[train])
                x_test = scaler.transform(X[test])

            elif Selected_Sclaer=='Standard_Scaler':
                scaler = preprocessing.StandardScaler().fit(X[train])
                x_train = scaler.transform(X[train])
                x_test = scaler.transform(X[test]) 

            elif Selected_Sclaer == 'No_Scaler':
                x_train = X[train]
                x_test = X[test]
            else:
                print('Scalling Method option was not correctly selected...!')

            ############Changes###############################

            prob = model.fit(x_train, y[train]).predict_proba(x_test)
            predicted = model.fit(x_train, y[train]).predict(x_test)
            fpr, tpr, thresholds = roc_curve(y[test], prob[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0

            TN, FP, FN, TP = confusion_matrix(y[test], predicted).ravel()

            TN = float(TN)
            FP = float(FP)
            FN = float(FN)
            TP = float(TP)       
             
            specificity = round(float(TN / (TN + FP)),2)
            sensitivity = round(float(TP / (TP + FN)),2)
            presison = round(float(TP /(TP + FP)),2)
            mcc =  round(matthews_corrcoef(y[test], predicted),2)           
            f1 =  round(f1_score(y[test], predicted),2)
            #ntr = round(float(TN/(TN+FN)),2)
            accuracy = round(float((TP+TN/(TP+TN+FP+FN))))

            specificity_list.append(specificity)
            sensitivity_list.append(sensitivity)
            presison_list.append(presison)
            mcc_list.append(mcc)
            f1_list.append(f1)

        spe_mean = float(sum(specificity_list))/float(len(specificity_list))
        sen_mean = float(sum(sensitivity_list))/float(len(sensitivity_list))
        pre_mean = float(sum(presison_list))/float(len(presison_list))
        mcc_mean = float(sum(mcc_list))/float(len(mcc_list))
        f1_mean = float(sum(f1_list))/float(len(f1_list))

        pl.plot([0, 1], [0, 1], '--', lw=2)
        mean_tpr /= folds.get_n_splits(X, y)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        pl.plot(mean_fpr, mean_tpr, '-', label='AUC = %0.2f' % mean_auc, lw=2)

        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('FP Rate',fontsize=22)
        pl.tick_params(axis='x', labelsize=22)
        pl.tick_params(axis='y', labelsize=22)
        pl.ylabel('TP Rate',fontsize=22)
        pl.legend(loc="lower right")
        pl.axis('tight')
        
        self.V_header = ("specificity","sensitivity","presison","mcc","f1")
        self.v_values = (round(spe_mean, 2),round(sen_mean, 2),round(pre_mean, 2),round(mcc_mean, 2),round(f1_mean, 2))
        mname  = ("Logistic_Regression","GaussianNB","KNeighbors","DecisionTree","SVC")

        pl.title(mname[self.st],fontsize=22)
        #pl.set_title(mname[self.st], fontsize=22)

        print (os.path.join(self.image_out_path,mname[self.st]+".png"))
        pl.savefig(os.path.join(self.image_out_path,mname[self.st]+".png"))
        pl.close()
        #pl.show()
        
        return self.V_header, self.v_values

    def main_pro(self, Des_Set, nfold, dir_path, out_file, report_file_name, report_file_dir_path, scaling_option):

        self.Des_Set = Des_Set
        self.nfold = nfold
        self.dir_path = dir_path
        self.out_file = out_file
        self.report_file_name = report_file_name
        self.report_file_dir_path = os.path.join(os.getcwd(),report_file_dir_path)
        self.scaling_option = scaling_option
        print (self.report_file_dir_path)


        if not os.path.exists(report_file_dir_path):
            os.makedirs(report_file_dir_path)
        
        val_list = []
        X, y = My_csv_class().data_gen(self.Des_Set)      
                
        LR = LogisticRegression(solver='lbfgs')
        GNB = GaussianNB()
        KNB = KNeighborsClassifier()
        DT = DecisionTreeClassifier()
        SV = SVC(probability=True,gamma='scale')
        
        classifiers = (LR, GNB, KNB, DT, SV)
       
        for ni, classifier in enumerate(classifiers):
           
            hdr, vlu = My_csv_class().roc_gen(X, y,classifier,self.nfold,ni,self.dir_path,self.report_file_dir_path,self.scaling_option)
            val_list.append(vlu)

        print (val_list)

        rdf = pd.DataFrame(val_list, index=("LR", "GNB", "KNB", "DT", "SV"), columns = hdr)  
        rdf.to_csv(os.path.join(self.dir_path,self.out_file),sep='\t')

        rdf.plot(kind='bar',label="Result Summary")
        pl.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4, prop={'size':11},ncol=5, mode="expand", borderaxespad=0.)
        pl.savefig(os.path.join(report_file_dir_path,'Result_summarty.png'))
        #pl.show()
        pl.close()



        out_html = open(os.path.join(report_file_dir_path, report_file_name),'w')              

        part_1 =  """

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

        div.table {
          width:600px;
          margin: auto;
         padding-right: 10; 

        }

        </style>
       </head>
        <div class="jumbotron text-center">
          <h1> Machine Learning Algorithm Assessment Report </h1>
        </div>
          


         <div class="container_1">
          <h2>Result Summary</h2>
          
          <table class="table">
            <thead>
              <tr class="danger">
                <th>Algorythm</th>
                <th>Specificity</th>
                <th>Sensitivity</th>
                <th>Presison</th>
                <th>MCC Score</th>
                <th>F1 Score</th>
              </tr>
            </thead>
            <tbody>
        """

        out_html.write(part_1)
          
        al = ["LR", "GNB", "KNB", "DT", "SV"]

        lines = pd.DataFrame.as_matrix(rdf)

        for j,line in enumerate(lines):
            out_html.write("\t\t<tr>\n")
            out_html.write("\t\t\t<td>"+str(al[j])+"</td>\n")
            for x,l in enumerate(line):
                #out_html.write("\t\t\t<td>"+str(al[x])+"</td>\n")
                out_html.write("\t\t\t<td>"+str(line[x])+"</td>\n")
            out_html.write("\t\t</tr>\n")

        part_3 = """
            </tbody>
          </table>
        </div>

        <div class="container">

          <h2>ROC curve and result summary Graph</h2>

          <div class="row">

            <div class="col-sm-4">
             
          
              <img src="DecisionTree.png" alt="Smiley face" height="350" width="350">
              <p>Lorem ipsum dolor sit amet, consectetur adipisicing elit...</p>
            
            </div>
            <div class="col-sm-4">
          
              <img src="GaussianNB.png" alt="Smiley face" height="350" width="350">
              <p>Lorem ipsum dolor sit amet, consectetur adipisicing elit...</p>

            </div>
            <div class="col-sm-4">
           
              <img src="KNeighbors.png" alt="Smiley face" height="350" width="350">
              <p>Lorem ipsum dolor sit amet, consectetur adipisicing elit...</p>
         
            </div>
                <div class="col-sm-4">
            
              <img src="Logistic_Regression.png" alt="Smiley face" height="350" width="350">
              <p>Lorem ipsum dolor sit amet, consectetur adipisicing elit...</p>
         
            </div>
            <div class="col-sm-4">
            
              <img src="SVC.png" alt="Smiley face" height="350" width="350">
              <p>Lorem ipsum dolor sit amet, consectetur adipisicing elit...</p>

            </div>
            <div class="col-sm-4">
              
              <img src="Result_summarty.png" alt="Smiley face" height="350" width="450">
              <p>Lorem ipsum dolor sit amet, consectetur adipisicing elit...</p>

            </div>
            
          </div>
        </div>

        </body>
        </html>
        """ 

        out_html.write(part_3)


        
if __name__=="__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-f", "--file_name",
                        required=True,
                        default=None,
                        help="Path to target CSV file")
                        
    parser.add_argument("-n", "--n_folds",
                        required=None,
                        default=5,
                        help="n_folds for Cross Validation")

    parser.add_argument("-w", "--Work_dir_path",
                        required=None,
                        default=os.getcwd(),
                        help="n_folds for Cross Validation")
                        
    parser.add_argument("-o", "--out_file_name",
                        required=True,
                        default=None,
                        help="Path to out file")
    
    parser.add_argument("-r", "--out__html_file_name",
                        required=None,
                        default='report_file.html',
                        help="Path to out file")

    parser.add_argument("-d", "--out_html_dir_name",
                required=None,
                default=os.path.join(os.getcwd(), 'report_dir'),   
                help="Path to out file")

    parser.add_argument("-s", "--Scalling_option",
                required=True,
                default=None,   
                help="for MinMaxScaler select option 'Min_Max' for StandardScaler select 'Standard_Scaler' and 'No_Scaler' for without scalling")
                       
    args = parser.parse_args()

    a = My_csv_class()
    a.main_pro(args.file_name, int(args.n_folds), args.Work_dir_path, args.out_file_name, args.out__html_file_name, args.out_html_dir_name,args.Scalling_option)



    

