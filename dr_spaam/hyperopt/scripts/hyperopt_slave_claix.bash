#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=hyperopt

#SBATCH --output=/home/yx643192/slurm_logs/hyperopt/%J_%x.log

#SBATCH --cpus-per-task=1

#SBATCH --mem-per-cpu=3G

#SBATCH --time=2-00:00:00

#SBATCH --signal=TERM@120

#SBATCH --partition=c18m

#SBATCH --account=rwth0485

#SBATCH --array=1-50

source $HOME/.zshrc
conda activate torch10

cd /work/yx643192/hyperopt_tmp

ssh -4 -N -f -J jia@recog.vision.rwth-aachen.de -L localhost:12345:chimay:27012 jia@chimay

PYTHONPATH=$PYTHONPATH:/home/yx643192/v3/hyperopt hyperopt-mongo-worker --mongo=localhost:12345/hyperopt --reserve-timeout=inf --poll-interval=15
