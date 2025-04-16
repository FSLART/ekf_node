from setuptools import find_packages, setup

package_name = 'ekf_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/ekf_node/launch', ['launch/launch_ekf.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tomas',
    maintainer_email='tomasmarcelinosantos@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ekf_node = ekf_node.state_estimator:main'
        ],
    },
)
