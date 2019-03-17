# python 2.7 
import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import count, avg

sc =SparkContext()
SparkContext.getOrCreate()
conf = SparkConf().setAppName("building a warehouse")
#sc = SparkContext(conf=conf)
sqlCtx = SQLContext(sc)
print ("==================")
print (sc)
print ("==================")

def run():
	df_train = sqlCtx.read.format('com.databricks.spark.csv')\
	                      .options(header='true', inferschema='true')\
	                      .load('/Users/yennanliu/NYC_Taxi_Trip_Duration/data/train.csv')
	df_group1= df_train.groupBy("store_and_fwd_flag", "vendor_id")\
	                   .agg(avg("pickup_longitude"), count("*"))
	df_group1.show()
	print (df_group1.show())
	#return df_group1
	#=====================
	df__ = sc.textFile("/Users/yennanliu/NYC_Taxi_Trip_Duration/data/train.csv")
	header = df__.first()
	df__value = df__.filter(lambda line: line != header)
	df__value.map(lambda x : x[0:][12:22]).take(10)
	print (df__value.map(lambda x : x[0:][12:22]).take(10))

def test():
	df_train = sqlCtx.read.format('com.databricks.spark.csv')\
	                      .options(header='true', inferschema='true')\
	                      .load('/Users/yennanliu/NYC_Taxi_Trip_Duration/data/train.csv')
	# select columns in DataFrame then transform to rdd 
	rdd_ = df_train.select('id','vendor_id','pickup_datetime').rdd
	type(rdd_)
	print ("==================")
	# id 
	print (rdd_.map(lambda x: x[0]).take(3))
	print ("==================")
	# vendor_id
	print(rdd_.map(lambda x: x[1]).take(3))
	print ("==================")
	# pickup_datetime
	print(rdd_.map(lambda x: x[2]).take(3))
	print ("==================")

def filter_column():
	# dataframe 
    df_train = sqlCtx.read.format('com.databricks.spark.csv')\
                      .options(header='true', inferschema='true')\
                      .load('/Users/yennanliu/NYC_Taxi_Trip_Duration/data/train.csv')
    # dataframe -> rdd 
    rdd_ = df_train.select('id','vendor_id','pickup_datetime').rdd
    # rdd -> dataframe
    df_xx = sqlCtx.createDataFrame(rdd_)
    # dataframe -> sql
    df_xx.registerTempTable("df_xx_table")
    sqlCtx.sql(""" 
				SELECT id, count(*) 
                FROM df_xx_table
                group by 1 
                order by 2 desc 
                limit 10""").show()

if __name__ == '__main__':
	#run()
	#test()
	filter_column()
