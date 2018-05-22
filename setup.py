from setuptools import setup, find_packages
with open('LICENSE') as f:
    license = f.read()

setup(name = 'etomo',
      version = '0.1',
      description = 'The toolbox for electron tomography.',
      author = 'Zhichao Zhong',
      author_email = 'zhong@cwi.nl',
      license = license,
      packages = find_packages( include = ('etomo',), exclude =('tests', 'example'))
      )

