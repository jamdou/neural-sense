@ECHO OFF
REM Run GPU profiles to work with nvvp
nvprof --quiet --export-profile C:\Users\Alex\University\NeuralSense\profile\timeline.prof python C:\Users\Alex\University\NeuralSense\main.py --profile=timeline
nvprof --quiet --metrics all --events all -o C:\Users\Alex\University\NeuralSense\profile\metricsEvents.prof python C:\Users\Alex\University\NeuralSense\main.py --profile=metric
python C:\Users\Alex\University\NeuralSense\main.py --profile=archive