# Jonathan McFadden
# Assignment 5

from pyspark import SparkConf, SparkContext
from pyspark.ml.clustering import KMeans

# from __future__ import print_function

import sys

import numpy as np
from pyspark.sql import SparkSession


spark = SparkSession\
        .builder\
        .appName("PythonKMeans")\
        .getOrCreate()

# conf = SparkConf()
# sc = SparkContext(conf=conf)

# dataset = sc.textFile("data.txt")
# dataset = spark.read.format

dataset = spark.read.format("libsvm").load("data.txt")

kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset)

wssse = model.computeCost(dataset)
print("Within set sum of squared errors = " + str(wssse))

centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)