## MixModel

### data format

- the input data should be a list
- each element in the list is a data for one patient
- each element is a tuple of three parts (LSTM) or 4 parts (TLSTM):
    1. static feature vector - a numpy array of all encoded static feature like demographic information (e.g., OHE) 
    2. seq feature vectors - a numpy array of numpy array shape (seq_len, feature_dim). Seq_len is number of encounters, feature dim is all features measure in an encounter. seq_len can be diff for each patient but feature_dim should be the same for all encounters and patient. (OHE, or numeric)
    3. time elapse - see TLSTM paper for how to define time interval (days from previous encounter). Our implementation did a modification: in TLSTM paper time elapse dim is (1, seq_len), we switch the dim to (seq_len, 1).
    4. label - binary ([0,1]or [1,0]) or multi-class (0,1,2,3...)
    
- after you prepared the data, save them as pickle file. Currently, we only support loading data via pickle.
- We do not support embeddings for mix model yet.
- example:
```
element data for TLSTM (LSTM format is same but no third time elapse data):

(
array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
        1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
        1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 0]),  # static feature (OHE in this case)
        
 array([[0.58845297, 0.15394126, 0.02184538, ..., 0.        , 0.02213439,
         0.        ],
        [0.58845297, 0.15394126, 0.02184538, ..., 0.        , 0.02213439,
         0.        ],
        [0.58845297, 0.15394126, 0.02184538, ..., 0.        , 0.02213439,
         0.        ],
        ...,
        [0.58845297, 0.15394126, 0.02184538, ..., 0.        , 0.02213439,
         0.        ],
        [0.58845297, 0.15394126, 0.02184538, ..., 0.        , 0.02213439,
         0.        ],
        [0.58845297, 0.15394126, 0.02184538, ..., 0.        , 0.02213439,
         0.        ]]),  # seq feature (numeric features in this case) (seq_len, feature_dim)
 
 array([[0.], [1.], [31.], [365.] ... [7.]]),  # time elapes (dim: (seq_len, 1))
        
 0 # label multiclass (use [1,0] in binary mode)
 )
``