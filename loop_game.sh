#!/bin/bash

while(true)
do
	java -jar Snake2017-alpha.jar -python client_agent1.py &
	sleep 300
        killall java
done
