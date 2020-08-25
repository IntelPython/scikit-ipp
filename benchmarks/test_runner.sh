pytest -vv --html=report.html --self-contained-html  --benchmark-group-by='param:input_dtype' --benchmark-columns='min, max, mean, stddev, median, rounds, iterations' --benchmark-json=results.json  test_benchmarks.py

pytest-benchmark compare results.json --csv=results.csv
