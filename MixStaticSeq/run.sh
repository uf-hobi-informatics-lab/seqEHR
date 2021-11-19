# # set up GPU
export CUDA_VISIBLE_DEVICES=-1

# # define data path
train_data='../data/train.pkl'
test_data='../data/test.pkl'
new_model='../temp/model'
res_output='../temp/result'
mlog='../temp/log.txt'

# if sequence length is different for each data point, set --various_seq_len flag
# if set --various_seq_len flag, batch size will be fixed at 1; consider using SGD optimizer

python task.py \
    --model_type lstm \
    --train_data_path $train_data \
    --test_data_path $test_data \
    --new_model_path $new_model  \
    --result_path $res_output \
    --do_train \
    --do_eval \
    --do_test \
    --optim adam \
    --learning_rate 1e-5 \
    --seed 13 \
    --dropout_rate 0.1 \
    --train_epochs 30 \
    --nonseq_hidden_dim 64 \
    --seq_hidden_dim 128 \
    --mix_hidden_dim 64 \
    --nonseq_representation_dim 64 \
    --mix_output_dim 2 \
    --log_step 2000 \
    --log_file $mlog \
    --batch_size 4 \
    --loss_mode bin
