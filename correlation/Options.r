  #############################################
 #                                             #
   # R script to preprocess  Descriptor Data #
 #                                             #
  #############################################
## library imported 
library(mlbench)
library(caret)
library(corrplot)
library(readr)
library(data.table)
library(argparser, quietly=TRUE)



p <- arg_parser("Round a floating point number")


p <- add_argument(p, "--Input_file", help="In put file" )
p <- add_argument(p, "--Output_file", help="Out put file")
p <- add_argument(p, "--CutOff", help="Coefficient CutOff", default=0.75)

# data from the file 

argv <- parse_args(p)
dat = read.csv(argv$Input_file, header = TRUE,sep = "\t")

# intial column count 

print ("intitial columns")
in_col = ncol(dat)
print(in_col)

# zero std column were removed 

nd = Filter(sd, dat)

#nd = dat[,colSums(dat != 0)> 0]
#Column count after removing zero value columns 

print ("After removing zero std value columsn")
print  (ncol(nd))

# Data normalization (zero mean, unit varience)

x =  ncol(nd)
preObj <- preProcess(nd[,2:(x-1) ], method=c("center", "scale"))
normalized_Data <- predict(preObj, nd[,2:(x-1)])

#z = ncol(normalized_Data)
#print (colMeans(normalized_Data[2:(z-1)]))
# correlation matrix 

y = ncol(normalized_Data)
m = cor(normalized_Data[,2:(y-1)])

#corelation plot 
#corrplot(m, method = "circle")
#removes Highly correlated columns  

hc = findCorrelation(m, cutoff=argv$CutOff)
new_hc = sort(hc)

########new_dat = normalized_Data[,-c(new_hc)] 
new_dat = dat[,-c(new_hc)] #Changed

#column count, Aftre removing the correlated columns . 
#z = ncol(new_dat)
#print (colMeans(new_dat[2:(z-1)]))

print ("after removing correlated columns")
print (ncol(new_dat))


#jai = ncol(new_dat)
# final data written to the csv file. 
#write.csv(new_dat, file = options[2], row.names = FALSE)

class_label = dat[in_col]
full_data_frame = cbind(new_dat)
fwrite(full_data_frame, argv$Output_file,sep = "\t")
