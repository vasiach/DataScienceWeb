from __future__ import division
import pandas as pd
import numpy as np
import cPickle as pickle


def load_data():
    # read raw data
    orders = pd.read_csv('data/orders.csv')
    products = pd.read_csv('data/products.csv')
    orders_priors = pd.read_csv('data/order_products__prior.csv')
    departments = pd.read_csv('data/departments.csv')
    aisles = pd.read_csv('data/aisles.csv')
    return orders, products, orders_priors, departments, aisles


def beverages_data(orders, products, orders_priors):
    # products in department 7
    products_bev = products[products['department_id'] == 7]
    products_bev_id = list(products_bev.product_id.values)

    orders_priors_bev = orders_priors[orders_priors['product_id'].isin(products_bev_id)]
    orders_prior_id_bev = list(orders_priors_bev.order_id.values)

    # orders of department 7
    orders_bev = orders[orders['order_id'].isin(orders_prior_id_bev)]
    return products_bev, products_bev_id, orders_bev, orders_prior_id_bev, orders_priors_bev


def bev_per_cart(orders_priors, products_bev_id, products, orders):

    # calculate beverages per cart
    #orders_priors['size_of_order'] = orders_priors.groupby('order_id')['add_to_cart_order'].transform('max')
    #orders_priors['total_beverages'] = orders_priors.groupby('order_id')['product_id'].transform(
     #   lambda x: (x.isin(products_bev_id)).sum())

    #orders_priors['bev_per_basket'] = np.where(orders_priors['total_beverages'] < 1, orders_priors['total_beverages'],
    #                                           np.around(
    #                                              orders_priors['total_beverages'] / orders_priors['size_of_order'],
     #                                             decimals=2))

    #orders_priors.to_pickle("orders_priors.pkl")
    orders_priors = pd.read_pickle("orders_priors.pkl")
    products_bev, products_bev_id, orders_bev, orders_prior_id_bev, orders_priors_bev = beverages_data(orders, products,
                                                                                                       orders_priors)
    orders_bev = orders_bev.merge(
        orders_priors_bev[['order_id', 'bev_per_basket']].drop_duplicates(subset=['order_id']), on='order_id',
        how='left')
    bev_per_basket = orders_bev.groupby('user_id')['bev_per_basket'].mean().reset_index()

    return bev_per_basket, products_bev, products_bev_id, orders_bev, orders_prior_id_bev, orders_priors_bev


def feature_extr(X_train):

    orders, products, orders_priors, departments, aisles = load_data()

    # convert categorical data to dummy values
    X_train = pd.get_dummies(X_train, prefix=["d", "h"], columns=['order_dow', 'order_hour_of_day'])

    products_bev, products_bev_id, orders_bev, orders_prior_id_bev, orders_priors_bev = beverages_data(orders, products, orders_priors)

    # count orders of department 7 for every user
    count_ord_bev = orders_bev.groupby('user_id').count()
    count_ord_bev = count_ord_bev['order_id']
    count_ord_bev = count_ord_bev.reset_index()
    count_ord_bev.rename(columns={'order_id': 'order_count'}, inplace=True)

    # merge feature 1 and fill nan values
    X_train = X_train.merge(count_ord_bev, on='user_id', how='left')
    X_train = X_train.fillna(0.0)

    # average interval days per user
    average_order_days = orders_bev.groupby('user_id')['days_since_prior_order'].mean().reset_index()
    average_order_days.rename(columns={'days_since_prior_order': 'avg_days_since_prior'}, inplace=True)

    # merge feature 2  and fill nan
    X_train = X_train.merge(average_order_days, on='user_id', how='left')
    X_train = X_train.fillna(0.0)

    bev_per_basket, products_bev, products_bev_id, orders_bev, orders_prior_id_bev, \
                                orders_priors_bev = bev_per_cart(orders_priors, products_bev_id, products, orders)

    # merge feature 3 and fill nan
    X_train = X_train.merge(bev_per_basket, on='user_id', how='left')
    X_train = X_train.fillna(0.0)

    return X_train

