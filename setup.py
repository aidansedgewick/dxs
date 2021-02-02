from setuptools import setup

setup(
        name="dxs",
        version="0.1.0",
        description="Run suite for DXS reduction",
        url="https://github.com/aidansedgewick/dxs",
        author="aidan-sedgewick",
        author_email='aidansedgewick@gmail.com',
        license="MIT license",
        #install_requires=requirements,
        #packages = find_packages(exclude=["docs"]),
)

from dxs.paths import create_all_paths

create_all_paths()

print("More setup is available in setup_scripts. See README for more.")
