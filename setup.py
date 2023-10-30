# -*- coding: utf-8 -*-
from setuptools import setup
import os

packages = \
['pipeline_dp',
 'pipeline_dp.dataset_histograms'
]

package_data = \
{'': ['*']}

install_requires = \
['numpy>=1.20.1,<2.0.0',
 'python-dp>=1.1.5rc4',
 'scipy>=1.7.3,<2.0.0'
]


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as fp:
        return fp.read()

setup_kwargs = {
    'name': 'pipeline-dp',
    'version': '0.2.1',
    'description': 'Framework for applying differential privacy to large datasets using batch processing systems',
    'author': 'Chinmay Shah',
    'author_email': 'chinmayshah3899@gmail.com',
    'maintainer': 'Vadym Doroshenko',
    'maintainer_email': 'dvadym@google.com',
    'url': 'https://github.com/OpenMined/PipelineDP/',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<3.12',
    'long_description': read("README.md"),
    'long_description_content_type': 'text/markdown',
    'license': "Apache-2.0",
}


setup(**setup_kwargs)
