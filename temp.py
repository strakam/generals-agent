import neptune

# Initialize Neptune run
run = neptune.init_project('strakam/supervised-agent')

# read run id SUP-216 and print from "matchups"

# get run
run = neptune.get_run('SUP-216')
print(run['matchups'])
