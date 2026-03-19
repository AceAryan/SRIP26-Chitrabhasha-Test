import pyarrow.parquet as pq
import pandas as pd

def load_stratified_subset(path, n_per_class=5000, seed=42):
    pf = pq.ParquetFile(path)
    chunks = []

    for batch in pf.iter_batches(batch_size=100_000, columns=["DATA", "TOPIC"]):
        chunk = batch.to_pandas()
        
        # Sample per group, keeping TOPIC column explicitly
        sampled_groups = []
        for topic, group in chunk.groupby("TOPIC"):
            n = min(len(group), 500)
            sampled_groups.append(group.sample(n, random_state=seed))
        
        chunks.append(pd.concat(sampled_groups, ignore_index=True))

    df = pd.concat(chunks, ignore_index=True)

    # Final downsample to n_per_class per topic
    final_groups = []
    for topic, group in df.groupby("TOPIC"):
        n = min(len(group), n_per_class)
        final_groups.append(group.sample(n, random_state=seed))

    return pd.concat(final_groups, ignore_index=True)


subset = load_stratified_subset("dataset_10M.parquet", n_per_class=5000)
print(subset["TOPIC"].value_counts())
print(f"Total rows: {len(subset)}")

## output

# TOPIC
# adult_content                  5000
# art_and_design                 5000
# crime_and_law                  5000
# education_and_jobs             5000
# electronics_and_hardare        5000
# entertainment                  5000
# fashion_and_beauty             5000
# finance_and_business           5000
# food_and_dining                5000
# games                          5000
# health                         5000
# history_and_geography          5000
# home_and_hobbies               5000
# industrial                     5000
# literature                     5000
# politics                       5000
# religion                       5000
# science_math_and_technology    5000
# social_life                    5000
# software                       5000
# software_development           5000
# sports_and_fitness             5000
# transportation                 5000
# travel_and_tourism             5000
# Name: count, dtype: int64
# Total rows: 120000