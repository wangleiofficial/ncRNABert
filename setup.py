from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='ncRNABert',
      version='0.1.1',
      description='ncRNA language model',
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords='ncRNA language model',
      classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
      url='https://github.com/wangleiofficial/ncRNABert',
      author='Lei Wang',
      author_email='wanglei@isyslab.org',
      license='MIT',
      packages=['ncRNABert'],
      install_requires=[
        'einops',
        'torch',
        'numpy',
        'rotary_embedding_torch'
    ],
      include_package_data=True,
      zip_safe=False)