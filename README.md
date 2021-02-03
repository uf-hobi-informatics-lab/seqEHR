# seqEHR
> develop deep learning-based models for learning structured EHR as sequence


## specific model desc

### TLSTM
> TLSTM first published in Patient Subtyping via Time-Aware LSTM Networks", KDD, 2017
- We re-implement the TLSTM using PyTorch in TLSTM
- the implementation was validated using the sync data released in the original TLSTM repo. 
- The results we got on train is AUC of 0.958 and test is AUC of 0.916
- input data format
```
# We use a slightly different input format compared to the original implementation.
# input - three conpoenents: 1. features (OHE or embeddings); 2. time intervals; 3. labels
# feature - 3D Tensor - [Batch, Sequence, Data]
# Time - 3D Tensor - [Batch, Sequence, time_diff]
# label - 2D numpy array - [Batch, label]

# Note: in original data released, the time diff data shape is [Batch, time_diff, Sequence] but we reshape this for dimension consistency
# see ./TLSTM/test_tlstm.py for more usage details
```

### LSTM
- This is baseline for comparison purpose
- input data shape
```
# feature - 3D Tensor - [Batch, Sequence, Data]
# label - 2D numpy array - [Batch, label]
```

### Embedding
- we only support pre-trained embeddings
- we do not support in-situ embedding random initialization
- if you want to random initialize embeddings, you have to create a random initialized embeddings yourself
- embedding input shape: ```[Batch, Squence, Features, Embeddings]```. Note that we have 4 dimensions instead of 3 because the each features has been converted to a embedding which is an dense vector whereas the OHE directy using 1-0 to represent features.

### self-attention (TODO)
- we implement a self-attention architecture to replace LSTM and TLSTM
- self-attention proved to be perform better in many NLP tasks over LSTM (seq2seq translation)
- position enmbedding can be used for encode time variance which is suitable for replace TLSTM

## MixStaticSeq model
- we develope a mix model which can handle both static features (e.g., demographics) and time-series features (e.g., diagnoses, medication, procedure, labs)
- we use MLP + TLSTM (or LSTM) as model architecture
- we have a specific data input format doc at: 
- The current evaluation will be measured as ROC-AUC, sensitivity, specificity, and accuracy
- input data format
```
# we expect the input data for MixStaticSeq model contains three parts:
# 1. static feature
# 2. sequence features - see TLSTM and LSTM for details
# 3. labels

# static features will be 2-d tensor as [Batch, features]
# 2 and 3 can be refered to TLSTM or LSTM based on your choice
```
