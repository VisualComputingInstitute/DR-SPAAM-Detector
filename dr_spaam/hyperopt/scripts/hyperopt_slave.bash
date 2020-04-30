#!/bin/bash
tmux -2 new-session -d -s slave_hyperopt

# tmux send-keys "ssh -L -f -N localhost:12345:chimay:27010" C-m

for (( c=1; c<$1; c++ ))
do
   tmux split-window -v
   tmux select-layout tiled
done


for (( c=0; c<$1; c++ ))
do
   tmux select-pane -t $c
   tmux send-keys "source /home/jia/torch_cuda10_venv/bin/activate" C-m
   tmux send-keys "cd /home/jia/tmp" C-m
   tmux send-keys "PYTHONPATH=$PYTHONPATH:~/drower9k hyperopt-mongo-worker --mongo=chimay:27010/hyperopt --reserve-timeout=inf --poll-interval=15" C-m
done

#Attach to session
tmux -2 attach-session -t slave_hyperopt
