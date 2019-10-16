from setuptools import setup, find_packages

setup(
    name='senseye-ml-challenge',
    description='Image segmentation problem',
    author='Author',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'opencv-python',
        'argparse',
        'matplotlib'
        ],

    package_data={
        '': ['*.save', '*.json', '*.h5']
   }
)