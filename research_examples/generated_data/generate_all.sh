#!/bin/sh
echo "Generating Data for Illustrate"
python3 data_generator.py --folder illustrate/ --illustrate

echo "Generating Data for basic Monte Carlo Experiments"
python3 data_generator.py --folder monte_carlo/ --n_files 100 --n_samples 1000

echo "Generating Data for unseen causality MC experiment"
python3 data_generator.py --folder unseen/ --unseen --n_files 100 --n_samples 1000

echo "Generating Data for Correlations MC experiment"
python3 data_generator.py --folder correlations/ --correlations --n_files 100 --n_samples 1000
