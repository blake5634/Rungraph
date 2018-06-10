
### Rungraph

A python program to analyze your data from [Mappedometer.com]{https://http://mappedometer.com}.

Mappedometer is a web based site which let's you set up routes on Google Maps, and enter your run times
(without needing to run with a phone or a GPS watch).

## Features

* Accumulates stats for the 16 runs you have logged the most
* Computes number of runs, and min/mean/max pace
* Displays your pace history, and pace histogram for any route.
* Compare your routes:
    * Displays a horizontal box plot of all your routes
    * Display your pace vs. elevation gain of the routes
    * Compare pace for your 3K vs 5K runs
    * Display a histogram of all your runs
* Determine how/if your pace correlates with your number of runs per week.
* Plot your minutes per week

## Usage

1. From your "Activity Log" in mappedometer, click "Export" and save as "ActivityLog.csv" (the default name)
2. >python rungr.py
