#!/bin/sh

timestamp=$(date +%Y%m%d%H%M%S)

#python snake_richard_tests_confusion.py | tee logs/snake_richard_tests_confusion_sphered_LR_Only_$timestamp.log

#python snake_richard_tests_confusion.py | tee logs/snake_richard_tests_confusion_sphered_SLR_$timestamp.log

#python snake_richard_tests_confusion_no_bn.py | tee logs/snake_richard_tests_confusion_no_bn_sphered_SLR_$timestamp.log


#python snake_richard_tests_generator_no_bn.py | tee logs/snake_richard_tests_generator_no_bn_sphered_SLR_$timestamp.log 


#python snake_richard_tests_generator.py | tee logs/snake_richard_tests_generator_sphered_LR_Only_$timestamp.log

#python snake_richard_tests_generator.py | tee logs/snake_richard_tests_generator_sphered_SLR_$timestamp.log

#python alexnet_main.py | tee log/alexnet_$timestamp.log

#tail -f logs/snake_richard_tests_$timestamp.log

#python generator_v0_1_1.py | tee logs/generator_v0_1_1_$timestamp.log

#python generator2_v0_1_1.py | tee logs/generator2_v0_1_1_$timestamp.log

#python generator_alexnet_v0_1_1.py| tee logs/generator_alexnet_v0_1_1_$timestamp.log

python generator_splitSLR_v0_1_1.py | tee logs/generator_splitSLR_v0_1_1_$timestamp.log
