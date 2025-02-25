## This script is intended to run a series of simulation scripts
# Specify the list of scripts below, and run this file

scripts = ["AstrometryRetrieval-ShotNoise1.py",
           "AstrometryRetrieval-ShotNoise2.py",
           "AstrometryRetrieval-ShotNoise3.py",
           "AstrometryRetrieval-ShotNoise4.py",
           "AstrometryRetrieval-ShotNoise5.py",
           "AstrometryRetrieval-ShotNoise6.py"]

for script in scripts:
    print(f"\nExecuting {script}...")
    with open(script) as f:
        exec(f.read())  # Runs the script as if it's part of this file
