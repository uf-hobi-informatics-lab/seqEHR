# seqEHR
> develop deep learning-based models for learning structured EHR as sequence

## MixStaticSeq model
- we develope a mix model which can handle both static features (e.g., demographics) and time-series features (e.g., diagnoses, medication, procedure, labs)
- we use MLP + TLSTM (or LSTM) as model architecture
- we have a specific data input format doc at: 
- The current evaluation will be measured as ROC-AUC, sensitivity, specificity, and accuracy

## specific model desc

### TLSTM
> TLSTM first published in Patient Subtyping via Time-Aware LSTM Networks", KDD, 2017
- We re-implement the TLSTM using PyTorch in TLSTM
- the implementation was validated using the sync data released in the original TLSTM repo. 
- The results we got on train is AUC of 0.958 and test is AUC of 0.916

### LSTM
- This is baseline for comparison purpose

### Embedding
- we only support pre-trained embeddings
- we do not support in-situ embedding random initialization
- if you want to random initialize embeddings, you have to create a random initialized embeddings yourself

### self-attention
- we implement a self-attention architecture to replace LSTM and TLSTM
- self-attention proved to be perform better in many NLP tasks over LSTM (seq2seq translation)
- position enmbedding can be used for encode time variance which is suitable for replace TLSTM