export JAVA_LD_LIBRARY_PATH='/afs/cs.pitt.edu/usr0/ras306/miniconda3/envs/psinet/jre/lib/amd64/server'
export JAVA_HOME='/afs/cs.pitt.edu/usr0/ras306/miniconda3/envs/psinet'
CUDA_VISIBLE_DEVICES=3 ./run.sh --ngpu 1 --feats_type raw --skip_data_prep false

