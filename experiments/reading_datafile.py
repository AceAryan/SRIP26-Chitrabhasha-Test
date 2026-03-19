import dask.dataframe as dd

df = dd.read_parquet("dataset_10M.parquet")
print(df.head())   