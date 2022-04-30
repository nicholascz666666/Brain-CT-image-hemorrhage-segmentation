import time
import datatable as dt
import polars


start = time.time()
dt.fread("subdural_window.csv")
end = time.time()
print(end - start)

start = time.time()
polars.read_csv("subdural_window.csv")
end = time.time()
print(end - start)
