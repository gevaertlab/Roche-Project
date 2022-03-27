## Methods to use RNA-Seq data from TCGA to predict cancer stage and grade

### Content
hr()
+ Cibersort
Codes to perform statics analysis and to generate violin plots for comparing immune cell types in different grades of prostate cancer. 

+ Gene_mode
Deep learning models to perform regression and classification tasks using RNA-Seq data. These methods use mini-batches and train-validation-test split to validate model, which does not involve cross validation. 


+ PASNet

Deep learning models that use PASNet to predict cancer types and the grade of prostate cancer. 
`Run_PASNet.py`: Run PASNet model with cross validation
`Run_Gene.py`: Run the raw gene classificiation model to benchmark the results with PASNet