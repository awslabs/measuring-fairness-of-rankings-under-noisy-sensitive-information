# InclusiveSearchFairnessMeasurement

** Describe InclusiveSearchFairnessMeasurement here **

## Documentation

Generated documentation for the latest released version can be accessed here:
https://devcentral.amazon.com/ac/brazil/package-master/package/go/documentation?name=InclusiveSearchFairnessMeasurement&interface=1.0&versionSet=live

## Cradle Jobs
Cradle scripts are ccurrently hed in the cradle directory. We have following cradle profiles implemented
- [FairnessMeasurement] Measuring Fairness in ranking: Dataset generation - https://datacentral.a2z.com/cradle/#/Alster/profiles/3ee3f7c0-ae9d-4404-8e82-24ede71cc8bc - cradle/fairness_measurement_dataset.py
  - This job can be used ot generate the data for fairness evaluation of Amazon search systems. Details in https://quip-amazon.com/kl9XAIzIwrj8/Measuring-Fairness-in-ranking
  
## Tesitng
When running from code root, use `PYTHONPATH=$(pwd)/test:$(pwd)/src:$PYTHONPATH python test/path2test_file` to run tests with the appropriate paths.

## CRADLE
1. Build the egg package with `python setup.py bdist_egg`
2. Find the egg file in `root/dist/InclusiveSearchFairnessMeasurement-1.0-{python-version}.egg`
3. Go to your cradle profile -> EDIT -> Show advanced settings
4. Use the Sideloaded Libraries dialog to upload thee egg build in p2. 

###NOTE - dependencies
If you are using nay additional libraries (e.g. numpy) you need to sideload the ZIP alongside your egg. For example the zip file for numpy can be downloaed from https://pypi.org/project/numpy/#files. Details in https://w.amazon.com/bin/view/BDT/Products/Cradle/Docs/PythonOnCradle/