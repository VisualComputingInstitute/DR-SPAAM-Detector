#!/bin/bash

# Loop over all panes, assuming their name is correct.
for machine in $(tmux list-windows -t hyperopt_master -F '#W'); do
    ssh $machine 'tmux kill-session -t slave_hyperopt'
done

# Finally kill the session
tmux kill-session -t hyperopt_master