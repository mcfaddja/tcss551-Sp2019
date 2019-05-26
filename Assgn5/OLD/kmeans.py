# Jonathan McFadden

import re
import sys
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)

lines = sc.textFile("data.txt")
dataset = lines.flatMap(lambda l: re.split(r' ', l))

print(dataset.collect())




sc.stop()
