pytest -vv --html=report.html --self-contained-html  --benchmark-group-by='param:input_dtype' --benchmark-columns='min, max, mean, stddev, median, rounds, iterations' --benchmark-json=results_warp.json  test_warp.py

pytest-benchmark compare results_warp.json --csv=results_warp.csv
