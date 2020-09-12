import pandas as pd
import numpy as np
from pathlib import Path

DATAPATH = Path("DATA/")

data = pd.read_csv(DATAPATH.joinpath("order_series.csv"))

user_ids = data.user_id.drop_duplicates()
user_ids = np.random.choice(user_ids, size = len(user_ids), replace = False)

train_users = user_ids[:int(0.8 * len(user_ids))]
val_users = user_ids[int(0.8 * len(user_ids)):int(0.9*len(user_ids))]
test_users = user_ids[int(0.9*len(user_ids)):]

data = data[['user_id', 'order_id', 'product_id']]
next_ = data.iloc[1:]
prior_ = data.iloc[:-1]

data = pd.DataFrame(np.column_stack((next_[['user_id', 'product_id']], prior_[['user_id', 'product_id']])), 
            columns = ["user_next","next_prod","user_prior","prior_prod"])

data = data[(data.user_next == data.user_prior)]
data = data[['user_next', 'next_prod', 'prior_prod']]
data.columns = ["user_id", "next", "prior"]

train_data = data[data.user_id.isin(train_users)]
val_data = data[data.user_id.isin(val_users)]
test_data = data[data.user_id.isin(test_users)]

train_data.to_csv(DATAPATH.joinpath("train.csv"))
val_data.to_csv(DATAPATH.joinpath("val.csv"))
test_data.to_csv(DATAPATH.joinpath("test.csv"))
