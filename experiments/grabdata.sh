#!/bin/bash
echo Grabbing data files from the robot

scp -r robot@192.168.0.50:/home/robot/robo_scripts/inference.csv /home/mi/Documents/uni/yr3/fyp/experiments/1sensorinference/

echo Done!
