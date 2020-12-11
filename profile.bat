@ECHO OFF

REM Run GPU profiles to work with nvvp

RMDIR .\profile /s /q
MKDIR .\profile

REM Run at full speed to produce a timeline
nvprof --quiet --export-profile .\profile\timeline.prof python . --profile=timeline

REM Run instruction level analysis
REM To run instruction level tests that also link to source code, you must enable debug mode ONLY for the kernel
REM that you want to test (nvidia only allows one for some reason). Be aware that this slows the execution of the
REM kernel
nvprof --quiet --source-level-analysis instruction_execution --export-profile .\profile\instructionLevel.prof python . --profile=instructionlevel

REM Run slowly and collects metrics for the smart report in nvvp
nvprof --quiet --metrics all --events all -o .\profile\metricsEvents.prof python . --profile=metric

REM Run python profiler
python -m cProfile -o .\profile\python.profile __main__.py

REM Archive the generated profiles
python . --profile=archive