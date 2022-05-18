"""Generate synthetic customer journeys for PipelineDP codelab."""

from absl import app
from absl import flags
import enum
from typing import Optional

import numpy as np
import pandas as pd

FLAGS = flags.FLAGS
flags.DEFINE_integer('n_customers', 100, 'The number of customers to simulate.')
flags.DEFINE_float('conversion_rate', .2, 'Conversion rate to simulate.')
flags.DEFINE_integer('random_seed', None, 'Random seed to use for simulations.')


class AvailableProducts(enum.IntEnum):
    """Class of available products."""
    JUMPER = 1
    T_SHIRT = 2
    SOCKS = 3
    JEANS = 4


_MINIMUM_PRICE = {'jumper': 40, 't_shirt': 20, 'socks': 5, 'jeans': 70}


class Product:
    """Class of products that can be viewed throughout a customer journey."""

    def __init__(self, product: AvailableProducts):
        self.name = product.name.lower()
        if self.name not in _MINIMUM_PRICE.keys():
            raise ValueError(
                f"{self.name} needs to be one of {_MINIMUM_PRICE.keys()}")
        self.minimum_price = _MINIMUM_PRICE[self.name]

    def cost(self,
             random_generator: Optional[np.random.Generator] = None) -> float:
        if not random_generator:
            random_generator = np.random.default_rng()
        return self.minimum_price + abs(np.round(random_generator.normal(), 2))


def create_customer_journeys(
        n_samples: int = 100,
        conversion_rate: float = .3,
        product_view_rate: float = .6,
        max_product_view: int = 5,
        random_generator: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """Creates synthetic data of customer product views and conversions.

  Args:
    n_samples: Number of samples to be generated.
    conversion_rate: Assumed conversion rate, i.e. probability that customer
      makes a purchase. Needs to be between 0-1.
    product_view_rate: Assumed probability that customer views a product. Needs
      to be between 0-1.
    max_product_view: Upper limit of possible product views. Needs to be >0. The
      expected number of viewed products is product_view_rate *
      max_product_view. For instance, if product_view_rate is .50 and
      max_product_view is 4, a customer will on minimum view two products.
    random_generator: Random generator that can be passed to make outputs
      reproducible.

  Returns:
    DataFrame of synthetic data.

  Raises:
    UserWarning: if either conversion_rate or product_view_rate is 0.
    ValueError: if max_product_view is 0.
  """
    all_customer_journeys = []

    if conversion_rate == 0 or product_view_rate == 0:
        raise UserWarning(
            'Setting conversion_rate or product_view_rate to 0 implies that no conversions can occur.'
        )

    if max_product_view <= 0:
        raise ValueError(
            f'max_product_view needs to be larger 0, but is {max_product_view}')

    if not random_generator:
        random_generator = np.random.default_rng()

    for _ in range(n_samples):
        n_products_viewed = np.sum(
            random_generator.binomial(1,
                                      p=product_view_rate,
                                      size=max_product_view))
        which_products_viewed = random_generator.integers(
            1, len(list(AvailableProducts)) + 1, size=n_products_viewed)
        is_conversion = random_generator.binomial(1, p=conversion_rate)
        products_viewed = {}
        basket_value = 0
        for index, product_id in enumerate(which_products_viewed):
            product = Product(product=AvailableProducts(product_id))
            products_viewed[f'product_view_{index}'] = product.name
            if is_conversion:
                basket_value += product.cost(random_generator=random_generator)

        products_viewed['conversion_value'] = basket_value
        all_customer_journeys.append(products_viewed)
    data = pd.DataFrame(all_customer_journeys)
    data.replace({'t_shirt': 't-shirt'}, inplace=True)
    return data.reindex(sorted(data.columns), axis=1)


def main(unused_argv):
    rng = np.random.default_rng(FLAGS.random_seed)
    df = create_customer_journeys(FLAGS.n_customers,
                                  conversion_rate=FLAGS.conversion_rate,
                                  random_generator=rng)

    df['has_conversion'] = (df['conversion_value'] > 0)
    df['user_id'] = df.index.values
    df.dropna(subset=['product_view_0'], inplace=True)
    df.fillna('none', inplace=True)
    df.to_csv('synthetic_customer_journeys.csv')


if __name__ == '__main__':
    app.run(main)
