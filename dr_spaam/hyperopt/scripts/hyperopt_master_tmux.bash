#!/bin/bash
tmux -2 new-session -d -s hyperopt_master

machines=(
  "Einhorn:12"
#  "Grimbergen:10"
#  "Bush:3"
#  "Carolus:2"
#  "Fix:2"
#  "Hund:2"
#  "Kriek:2"
#  "Schlunz:2"
#  "Tsingtao:4"
#  "Veltins:2"
#  "Zhiguli:2"
#  "Astra:1"
 # "Faxe:2"
#  "Grolsch:2"
#  "Hoppiness:6"
#  "Kilkenny:4"
#  "Lasko:4"
#  "Mickey:6"
#  "Paulaner:2"
#  "Bevog:2"
#  "Borsodi:2"    <-- dies with hp workers.
#  "Duff:2"
#  "Duvel:3"
#  "Helios:3"
#  "Tyskie:2"
#  "Reissdorf:8"
#  "Becks:2"
#  "Corona:4"
#  "Kingfisher:4"
#  "Stella:2"
  "Chimay:10"
#  "Rothaus:2"
)

# Create the windows for each machine.
for m in ${machines[@]}; do
  machine=${m%:*}
  count=${m#*:}

  echo $machine:$count
  tmux rename-window "$machine"
  tmux new-window
done

# Fix the redundant window created by the last loop entry.
# And move to the first window again.
tmux kill-window
sleep 1

# SSH to the actual machine and run the jobs
for m in ${machines[@]}; do
  machine=${m%:*}
  count=${m#*:}
  tmux send-keys -t hyperopt_master:$machine "ssh $machine" C-m
  tmux send-keys -t hyperopt_master:$machine  "~/drower9k/hyperopt_scripts/hyperopt_slave.bash $count" C-m
done

tmux select-window -t hyperopt_master:0

#Attach to session
tmux -2 attach-session -t hyperopt_master
