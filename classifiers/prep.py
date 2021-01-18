import os
import csv

rootdir = "./text_data/"

categories = {
    "business": "1",
    "entertainment": "2",
    "politics": "3",
    "sport": "4",
    "tech": "5",
}

rows = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        id_ = subdir.split("/")[-1]
        category = categories.get(id_)
        if category:
            path = os.path.join(subdir, file)
            with open(path) as text_data:
                text = text_data.read()
                rows.append((text, category))

with open("./text_data/data.csv", "w") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerows(rows)

