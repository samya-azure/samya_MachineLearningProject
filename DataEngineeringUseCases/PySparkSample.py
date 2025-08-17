
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr

# Step 1: Create Spark session
spark = SparkSession.builder.appName("StockDifferenceAnalysis").getOrCreate()

# Step 2: Load snapshot and current stock data
snapshot_df = spark.read.csv("./DataSets/snapshot_stock.csv", header=True, inferSchema=True)
current_df = spark.read.csv("./DataSets/current_stock.csv", header=True, inferSchema=True)

# Step 3: Join on item_code to compare
joined_df = snapshot_df.alias("snap").join(
    current_df.alias("curr"),
    on="item_code"
).select(
    col("item_code"),
    col("snap.item_name").alias("item_name"),
    col("snap.quantity").alias("yesterday_qty"),
    col("curr.quantity").alias("today_qty")
)

# Step 4: Add column showing difference
diff_df = joined_df.withColumn("difference", col("today_qty") - col("yesterday_qty"))

# Step 5: Filter items below reorder threshold (say < 40 units)
alert_df = diff_df.filter(col("today_qty") < 40)

# Step 6: Show results
print("Stock Difference Report:")
diff_df.show()

print("Items Below Reorder Level:")
alert_df.show()

