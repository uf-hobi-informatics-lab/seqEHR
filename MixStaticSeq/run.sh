# # set up GPU
export CUDA_VISIBLE_DEVICES=-1

# # define data path
train_data='../data/train.pkl'
test_data='../data/test.pkl'
new_model='./model'
res_output='./result'
mlog='./log.txt'

# # run experiment
# # train and test
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
  --learning_rate 1e-3 \
  --dropout_rate 0.1 \
  --train_epochs 20 \
  --do_warmup \
  --warmup_ratio 0.1 \
  --nonseq_hidden_dim 64 \
  --seq_hidden_dim 64 \
  --mix_hidden_dim 64 \
  --nonseq_representation_dim 64 \
  --mix_output_dim 2 \
  --loss_mode bin

# # only test
#python task.py \
#  --model_type clstm \
#  --train_data_path $train_data \
#  --test_data_path $test_data \
#  --new_model_path $new_model  \
#  --result_path $res_output \
#  --do_eval \
#  --do_test \
#  --optim adam \
#  --learning_rate 1e-3 \
#  --dropout_rate 0.1 \
#  --train_epochs 20 \
#  --do_warmup \
#  --warmup_ratio 0.1 \
#  --nonseq_hidden_dim 64 \
#  --seq_hidden_dim 64 \
#  --mix_hidden_dim 64 \
#  --nonseq_representation_dim 64 \
#  --mix_output_dim 2 \
#  --loss_mode bin