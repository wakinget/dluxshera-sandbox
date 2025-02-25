## This script is intended to run a series of simulation scripts
# Specify the list of scripts below, and run this file

scripts = ["AstrometryRetrieval-DefocusTest1.py",
           "AstrometryRetrieval-DefocusTest2.py",
           "AstrometryRetrieval-DefocusTest3.py"]

for script in scripts:
    print(f"\nExecuting {script}...")
    with open(script) as f:
        exec(f.read())  # Runs the script as if it's part of this file
