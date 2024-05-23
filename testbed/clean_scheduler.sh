#!/bin/bash

ps -ef | grep scheduler.py | grep -v grep
ps -ef | grep scheduler.py | grep -v grep | awk '{print $2}' | xargs kill
