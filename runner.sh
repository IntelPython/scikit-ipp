pytest -vv --html=report.html --self-contained-html  --benchmark-group-by='param:input_dtype' --benchmark-columns='min, max, mean, stddev, median, rounds, iterations' --benchmark-json=results_rotate.json  benchmarks/test_rotate.py

pytest-benchmark compare results_rotate.json --csv=results_rotate.csv
