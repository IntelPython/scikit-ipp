# scikit-ipp/benchmarks

## Example for running benhcmark tests

Running `test_rotate.py`

```
pytest -vv --html=report.html --self-contained-html  --benchmark-group-by='param:input_dtype' --benchmark-columns='min, max, mean, stddev, median, rounds, iterations' --benchmark-json=results_rotate.json  test_rotate.py

pytest-benchmark compare results_rotate.json --csv=results_rotate.csv
```

### TODO
* html or csv reporter for benchmark tests
* Runner for benchmark tests
