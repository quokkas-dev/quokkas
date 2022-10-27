from setuptools import setup, find_packages

with open('requirements.txt') as input:
    requirements = input.read()

setup(
    name='quokkas',
    version='0.0.2',
    description="Data analysis tool that you didn't know you needed",
    author='Ivan Pashkevich, Roman Sorokin',
    author_email='ivan.pashkevich@gmail.com, sorokin.r.v.97@gmail.com',
    python_requires='>=3.6',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements.splitlines()
)
