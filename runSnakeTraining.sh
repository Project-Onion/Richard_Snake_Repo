#!/bin/sh

timestamp=$(date +%Y%m%d%H%M%S)

#python snake_richard_tests_confusion.py | tee logs/snake_richard_tests_confusion_sphered_LR_Only_$timestamp.log

#python snake_richard_tests_confusion.py | tee logs/snake_richard_tests_confusion_sphered_SLR_$timestamp.log

#python snake_richard_tests_generator.py | tee logs/snake_richard_tests_generator_sphered_LR_Only_$timestamp.log

python snake_richard_tests_generator.py | tee logs/snake_richard_tests_generator_sphered_SLR_$timestamp.log

#python alexnet_main.py | tee log/alexnet_$timestamp.log

#tail -f logs/snake_richard_tests_$timestamp.log
