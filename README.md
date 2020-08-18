# seqEHR
> develop deep learning-based models for learning structured EHR as sequence

## TLSTM
- this is a PyTorch implementation of TLSTM published in Patient Subtyping via Time-Aware LSTM Networks", KDD, 2017
- the implementation was validated using the sync data released in the original TLSTM repo. 
- The results we got on train is AUC of 0.958 and test is AUC of 0.916

## TLSTMgeneral
- an update verions of TLSTM for general using purpose.
- the input file should be named as follow: features.pkl; times.pkl; lables.pkl
- features: OHE encoded EHR features
- times: elapsed relative times (see TLSTM paper)
- labels: [0,1] or [1,0]
- to train model with both OHE and numeric feature, you have to use the ensemble model (flag: ensemble)