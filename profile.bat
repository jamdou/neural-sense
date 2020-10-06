@ECHO OFF
REM Run GPU profiles to work with nvvp
RMDIR .\profile /s /q
MKDIR .\profile
nvprof --quiet --export-profile .\profile\timeline.prof python .\main.py --profile=timeline
nvprof --quiet --metrics all --events all -o .\profile\metricsEvents.prof python .\main.py --profile=metric
python .\main.py --profile=archive