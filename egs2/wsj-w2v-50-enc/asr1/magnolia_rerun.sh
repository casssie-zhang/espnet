export JAVA_LD_LIBRARY_PATH='/afs/cs.pitt.edu/usr0/ras306/miniconda3/envs/psinet/jre/lib/amd64/server'
export JAVA_HOME='/afs/cs.pitt.edu/usr0/ras306/miniconda3/envs/psinet'
CUDA_VISIBLE_DEVICES=2 ./exp/asr_train_asr_transformer_wav2vec_encoder_raw_char/run.sh
