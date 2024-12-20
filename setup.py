import os
from setuptools import setup, find_packages
print(os.getcwd())
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()
    print('long_description',long_description)
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    print('here',here)
    print(__file__)
    print(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-
    print(os.path.join(here, rel_path))
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()
print('yes',read('./llm_attacks/__init__.py').splitlines())
print('find_packages()',find_packages())
def get_version(rel_path):

    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            return line.split("'")[1]

    raise RuntimeError('Unable to find version string.')


with open('requirements.txt', 'r') as requirements:
    # print(os.listdir('./llm_attacks'))
    # print(os.listdir('llm_attacks'))
    # print(find_packages())
    setup(name='llm_attacks',
          version=get_version('./llm_attacks/__init__.py'),
          install_requires=list(requirements.read().splitlines()),
          packages=find_packages(),
          description='library for creating adversarial prompts for language models',
          python_requires='>=3.6',
          author='Andy Zou, Zifan Wang, Matt Fredrikson, J. Zico Kolter',
          author_email='jzou4@andrew.cmu.edu',
          classifiers=[
              'Programming Language :: Python :: 3',
              'License :: OSI Approved :: MIT License',
              'Operating System :: OS Independent'
          ],
          long_description=long_description,
          long_description_content_type='text/markdown')