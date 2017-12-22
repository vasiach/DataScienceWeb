from __future__ import division
import pandas as pd
import numpy as np


def feature_extr(X_train):
    # read raw data
    orders = pd.read_csv('data/orders.csv')
    products = pd.read_csv('data/products.csv')
    orders_priors = pd.read_csv('data/order_products__prior.csv')
    departments = pd.read_csv('data/departments.csv')
    aisles = pd.read_csv('data/aisles.csv')

    # convert categorical data to dummy values
    X_train_dum = pd.get_dummies(X_train, prefix=["d", "h"], columns=['order_dow', 'order_hour_of_day'])

    # products in department 7
    products_bev = products[products['department_id'] == 7]
    products_bev_id = list(products_bev.product_id.values)

    orders_priors_bev = orders_priors[orders_priors['product_id'].isin(products_bev_id)]
    orders_prior_id_bev = list(orders_priors_bev.order_id.values)

    # orders of department 7
    orders_bev = orders[orders['order_id'].isin(orders_prior_id_bev)]

    # count orders of department 7 for every user
    count_ord_bev = orders_bev.groupby('user_id').count()
    count_ord_bev = count_ord_bev['order_id']
    count_ord_bev = count_ord_bev.reset_index()
    count_ord_bev.rename(columns={'order_id': 'order_count'}, inplace=True)

    # merge feature 1
    X_train = X_train.merge(count_ord_bev, on='user_id', how='left')
    # Fill naan values
    X_train = X_train.fillna(0.0)

    # calculate beverages per order
    orders_priors['size_of_order'] = orders_priors.groupby('order_id')['add_to_cart_order'].transform('max')
    orders_priors['total_beverages'] = orders_priors.groupby('order_id')['product_id'].transform(
        lambda x: (x.isin(products_bev_id)).sum())

    orders_priors['bev_per_basket'] = np.where(orders_priors['total_beverages'] < 1, orders_priors['total_beverages'],
                                               np.around(
                                                   orders_priors['total_beverages'] / orders_priors['size_of_order'],
                                                   decimals=2))
    orders = orders.merge(orders_priors, how='left', on='order_id')
    orders.drop(columns=['product_id', 'add_to_cart_order', 'reordered', 'size_of_order', 'total_beverages'])

    return X_train

