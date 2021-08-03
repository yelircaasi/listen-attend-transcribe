import sys
import os
from pathlib import Path
#import data

sys.path.append(os.path.join(sys.path[0].rsplit("/", 1)[0], "src"))
#print(sys.path)

import data
#print(dir(data))

if len(sys.argv) > 1:
    DATA_DIR = sys.argv[1]
else:
    DATA_DIR = "/mount/studenten/arbeitsdaten-studenten1/rileyic"

# timit
#data.inspect_data(DATA_DIR, "timit", "train", 3, "phones", 1)
#data.inspect_data(DATA_DIR, "timit", "test", 3, "phones", 1)
#data.inspect_data(DATA_DIR, "timit", "dev", 3, "phones", 1)
#data.inspect_data(DATA_DIR, "timit", "train", 3, "bin", 2)
#data.inspect_data(DATA_DIR, "timit", "test", 3, "bin", 2)
#data.inspect_data(DATA_DIR, "timit", "dev", 3, "bin", 3)
#data.inspect_data(DATA_DIR, "timit", "train", 3, "cont", 3)
#data.inspect_data(DATA_DIR, "timit", "test", 3, "cont", 4)
#data.inspect_data(DATA_DIR, "timit", "dev", 3, "cont", 4)

# arcticl2
data.inspect_data(DATA_DIR, "arcticl2_all", "train", 3, "phones", 1)
#data.inspect_data(DATA_DIR, "arcticl2_all", "test", 3, "phones", 1)
#data.inspect_data(DATA_DIR, "arcticl2_all", "dev", 3, "phones", 1)
#data.inspect_data(DATA_DIR, "arcticl2_all", "train", 3, "bin", 2)
#data.inspect_data(DATA_DIR, "arcticl2_all", "test", 3, "bin", 2)
#data.inspect_data(DATA_DIR, "arcticl2_all", "dev", 3, "bin", 3)
#data.inspect_data(DATA_DIR, "arcticl2_all", "train", 3, "cont", 3)
#data.inspect_data(DATA_DIR, "arcticl2_all", "test", 3, "cont", 4)
#data.inspect_data(DATA_DIR, "arcticl2_all", "dev", 3, "cont", 4)

# arabicsc
data.inspect_data(DATA_DIR, "arabicsc", "train", 3, "phones", 1)
#data.inspect_data(DATA_DIR, "arabicsc", "test", 3, "phones", 1)
#data.inspect_data(DATA_DIR, "arabicsc", "dev", 3, "phones", 1)
#data.inspect_data(DATA_DIR, "arabicsc", "train", 3, "bin", 2)
#data.inspect_data(DATA_DIR, "arabicsc", "test", 3, "bin", 2)
#data.inspect_data(DATA_DIR, "arabicsc", "dev", 3, "bin", 3)
#data.inspect_data(DATA_DIR, "arabicsc", "train", 3, "cont", 3)
#data.inspect_data(DATA_DIR, "arabicsc", "test", 3, "cont", 4)
#data.inspect_data(DATA_DIR, "arabicsc", "dev", 3, "cont", 4)

# buckeye
data.inspect_data(DATA_DIR, "buckeye", "train", 3, "phones", 1)
#data.inspect_data(DATA_DIR, "buckeye", "test", 3, "phones", 1)
#data.inspect_data(DATA_DIR, "buckeye", "dev", 3, "phones", 1)
#data.inspect_data(DATA_DIR, "buckeye", "train", 3, "bin", 2)
#data.inspect_data(DATA_DIR, "buckeye", "test", 3, "bin", 2)
#data.inspect_data(DATA_DIR, "buckeye", "dev", 3, "bin", 3)
#data.inspect_data(DATA_DIR, "buckeye", "train", 3, "cont", 3)
#data.inspect_data(DATA_DIR, "buckeye", "test", 3, "cont", 4)
#data.inspect_data(DATA_DIR, "buckeye", "dev", 3, "cont", 4)


""" !!!
https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
"""