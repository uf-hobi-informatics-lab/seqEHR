# seqEHR
> develop deep learning-based models for learning structured EHR as sequence

## TLSTM
> TLSTM first published in Patient Subtyping via Time-Aware LSTM Networks", KDD, 2017
- We re-implement the TLSTM using PyTorch in TLSTM
- the implementation was validated using the sync data released in the original TLSTM repo. 
- The results we got on train is AUC of 0.958 and test is AUC of 0.916

## LSTM
- we use LSTM as baseline compared to TLSTM

## mix features
- In EHR, features can be sequence-like (e.g., diagnosis, medication) or non-sequence-like (e.g, demographics)
- we implement a mix feature model to handle both two types of features