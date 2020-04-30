d#!/bin/bash
tmux -2 new-session -d -s hyperopt_mongodb

tmux send-keys "mongod --dbpath /home/jia/tmp/dumps/drow/hyperopt_mongodb --port 27012 --directoryperdb --journal --bind_ip_all" C-m

#Attach to session
tmux -2 attach-session -t hyperopt_mongodb
