#!/usr/bin/env bash
make clean build

./thrustAlgorithms 100 transform > output.txt
./thrustAlgorithms 100 reduction >> output.txt
./thrustAlgorithms 100 prefix-sum >> output.txt
./thrustAlgorithms 100 reordering >> output.txt
./thrustAlgorithms 100 sorting >> output.txt
