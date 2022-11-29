from setuptools import setup, find_packages

with open('requirements.txt') as input:
    requirements = input.read()

with open('README.md') as input:
    readme = input.read()


setup(
    name='quokkas',
    version='0.0.4',
    description="Data analysis tool that you didn't know you needed",
    author='Ivan Pashkevich, Roman Sorokin',
    author_email='ivan.ig.pashkevich@gmail.com, sorokin.r.v.97@gmail.com',
    python_requires='>=3.7',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements.splitlines(),
    long_description=readme,
    long_description_content_type='text/markdown'
)
