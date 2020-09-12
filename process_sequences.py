# TODO: Add expanding window to the prior sequence, with a time since tagging.

import pandas as pd
import numpy as np
from pathlib import Path
from functools import reduce

DATAPATH = Path("DATA")

orders = pd.read_csv(DATAPATH.joinpath("orders/orders.csv"))
prior_orders = pd.read_csv(DATAPATH.joinpath("order_products__prior/order_products__prior.csv"))
products = pd.read_csv(DATAPATH.joinpath("products/products.csv"))

EOS_ID = products.product_id.max() + 1
SOS_ID  = EOS_ID + 1
VOCAB_SIZE = SOS_ID + 1

orders = orders[['order_id', 'user_id', 'order_number']].merge(prior_orders, on= 'order_id', how = 'inner')
aggregated_orders = orders.groupby(["user_id", "order_id", "order_number"]).product_id\
        .apply(lambda l: reduce((lambda x,y: str(x) + '_' + str(y)), l))

aggregated_orders = aggregated_orders.map(lambda x: str(SOS_ID) + '_' + str(x) + '_' + str(EOS_ID))
aggregated_orders.reset_index(drop = False).to_csv(DATAPATH.joinpath("order_series.csv"))

parameters = [  ['EOS_ID', EOS_ID],
                ['SOS_ID', SOS_ID],
                ['VOCAB_SIZE', VOCAB_SIZE]
                ]
parameters = pd.DataFrame(parameters).to_csv("vocab_params.csv")

