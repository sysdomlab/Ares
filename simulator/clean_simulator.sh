#!/bin/bash

ps -ef | grep start_for_full_simulation.py | grep -v grep
ps -ef | grep start_for_full_simulation.py | grep -v grep | awk '{print $2}' | xargs kill -9

ps -ef | grep simulator.py | grep -v grep
ps -ef | grep simulator.py | grep -v grep | awk '{print $2}' | xargs kill -9
