from setuptools import setup, find_packages
with open('LICENSE') as f:
    license = f.read()

setup(name = 'astra-et',
      version = '0.1',
      description = 'The astra toolkit for electron tomography.',
      author = 'Zhichao Zhong',
      author_email = 'zhong@cwi.nl',
      license = license,
      packages = find_packages( include = ('astraet',), exclude =('tests','data', 'samples'))
      )

