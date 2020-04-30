#!/bin/bash

cd /tmp
for (( c=0; c<24; c++ ))
do
   PYTHONPATH=$PYTHONPATH:~/drower9k hyperopt-mongo-worker --mongo=einhorn:27010/hyperopt --reserve-timeout=inf --poll-interval=15 &
done
wait %1
