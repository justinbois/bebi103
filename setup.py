from setuptools import setup, find_packages
from codecs import open
from os import path
import json

def read(fname):
    with open(fname, mode = "r", encoding = "utf-8") as f:
        src = f.read()
    return src

codemeta_json = "codemeta.json"

# Let's pickup as much metadata as we need from codemeta.json
with open(codemeta_json, mode = "r", encoding = "utf-8") as f:
    src = f.read()
    meta = json.loads(src)

# Let's make our symvar string
__version__ = meta["version"]

# Now we need to pull and format our author, author_email strings.
author = ""
author_email = ""
for obj in meta["author"]:
    given = obj["givenName"]
    family = obj["familyName"]
    email = obj["email"]
    if len(author) == 0:
        author = given + " " + family
    else:
        author = author + ", " + given + " " + family
    if len(author_email) == 0:
        author_email = email
    else:
        author_email = author_email + ", " + email
description = meta['description']
url = meta['codeRepository']
license = meta['license']
name = meta['name']
keywords= meta['keywords']

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name=name,
    version=__version__,
    description=description,
    long_description=long_description,
    url=url,
    download_url='https://github.com/justinbois/bebi103/tarball/' + __version__,
    license='BSD',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords=keywords,
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    author=author,
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email=author_email
)
