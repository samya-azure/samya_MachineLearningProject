
#After running the script, open your browser to http://localhost:4040

#Go to the “Stages” and “Tasks” tab to see each partition task

from pyspark.sql import SparkSession

# Step 1: Start Spark Session with all local cores
spark = SparkSession.builder \
    .appName("PartitionDemo") \
    .master("local[*]") \
    .getOrCreate()

# Step 2: Create a sample DataFrame with 1 million numbers and 4 partitions
data = spark.range(0, 1000000, numPartitions=4)

# Step 3: Show how many partitions are used
print(f"Number of partitions: {data.rdd.getNumPartitions()}")

# Step 4: Define a function to show partition-wise data
def show_partition_data(index, iterator):
    yield f"Partition {index} contains {len(list(iterator))} records"

# Step 5: Use mapPartitionsWithIndex to print how data is split
partition_data = data.rdd.mapPartitionsWithIndex(show_partition_data)
for line in partition_data.collect():
    print(line)

# Step 6: Stop the session
spark.stop()
