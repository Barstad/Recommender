#%%
import pandas as pd
import numpy as np
from pathlib import Path

DATAPATH = Path('DATA')

orders = pd.read_csv(DATAPATH.joinpath("orders/orders.csv"))
orders = orders[['user_id', 'order_id', 'order_number', 'days_since_prior_order']]

prior_products = pd.read_csv(DATAPATH.joinpath('order_products__prior/order_products__prior.csv'))
train_products = pd.read_csv(DATAPATH.joinpath('order_products__train/order_products__train.csv'))

if not DATAPATH.joinpath("order_content.csv").exists():
    order_sequences = prior_products.groupby("order_id").product_id\
        .apply(lambda l: reduce((lambda x,y: str(x) + '_' + str(y)), l))

    order_sequences.to_csv(DATAPATH.joinpath("order_content.csv"))

order_sequences = pd.read_csv(DATAPATH.joinpath("order_content.csv"))

#%%
orders["total_days_since"] = orders.sort_values(['user_id', 'order_number'], ascending = [True, False])\
            .groupby(["user_id"]).days_since_prior_order.apply(np.cumsum).shift(1).fillna(0).astype(int)
orders = orders.merge(order_sequences, how = 'inner', on = 'order_id')

#%%
import time
order_list = np.unique(orders.order_id)

order_id = 2539329
def get_features_from_order_id(order_id):
    example = orders[orders.order_id == order_id]
    user_id = example.user_id.values[0]
    order_number = example.order_number.values[0]
    days_since = example.total_days_since.values[0]

    prior = orders.loc[(orders.order_number < order_number) & (orders.user_id == user_id)]
    prior = prior.copy()
    prior.total_days_since -= days_since

    prior['basket_size'] = prior.product_id.apply(lambda x: len(x.split('_')))
    prior["days_since_str"] = prior.apply(lambda x: '_'.join([str(x.total_days_since)] * x.basket_size), axis = 1)
    products = prior.product_id.str.cat(sep = '_')
    timing = prior.days_since_str.str.cat(sep = '_')
    y = example.product_id.values[0]
    return [products, timing, y]

get_features_from_order_id(2254736)

#%%
orders.head()
#%%
import csv

with open('dataset.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(["product_history", "days_since", "next_order"])
    for i in order_list:
        try:
            row_list = get_features_from_order_id(i)
        except:
            continue
        writer.writerow(row_list)

#%%
# Adding next order
a = orders.iloc[:-1].reset_index(drop=True)
b = orders.iloc[1:].reset_index(drop = True)[['user_id', 'order_id']]
b.columns = ['user_id2', 'next_order_id']
orders = pd.concat([a,b], axis = 1)

# Remove overlapping users
orders = orders[orders.user_id == orders.user_id2]
orders = orders.drop('user_id2', axis = 1)
#%%
#%%
orders

#%%
full_orders = prior_products.merge(orders, how = 'inner', on = "order_id")

#%%
full_orders = full_orders.sort_values(["user_id", "order_number"], ascending = ["True", "True"])
# %%
full_orders.head()
#%%
current_user = 0
current_order_id = 0
days_since = 0
total_days_since = 0
dataset = []

for i in full_orders.iloc[:100].itertuples():
    if current_user != i.user_id:
        current_user = i.user_id
        prod_string = ""
        days_since = 0
        total_days_since = 0

    if current_order_id != i.order_id:
        current_order_id = i.order_id

    


    
    # days_since += 
    
    # y = i.next_order
    # x = [i.user_id, i.order_id, i.order_number, days_since, total_days_since]
    
# %%
