from pyspark.sql import SparkSession

# 初始化SparkSession
spark = SparkSession.builder \
    .appName("GLUE_Preprocessing") \
    .getOrCreate()

# 真实路径 (使用原始字符串r防止转义)
train_path = r"C:\Users\guanjiaqi\PyCharmMiscProject\glue\SST-2\train.tsv"
dev_path = r"C:\Users\guanjiaqi\PyCharmMiscProject\glue\SST-2\dev.tsv"

# 读取GLUE数据
train_df = spark.read.csv(train_path, sep='\t', header=True)
dev_df = spark.read.csv(dev_path, sep='\t', header=True)

print("训练集大小:", train_df.count())
train_df.show(5)

print("验证集大小:", dev_df.count())
dev_df.show(5)

spark.stop()
train_pd = train_df.toPandas()
dev_pd = dev_df.toPandas()
