from setuptools import find_packages, setup

package_name = 'turtle_tag'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aaron Serebrenik, Gábor Eszter',
    maintainer_email='yhc52f@inf.elte.hu, nhlyn2@inf.elte.hu',
    description='MARL Tag game with ROS2 and Pettingzoo',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'turtle_tag = turtle_tag.simple_tag:main',
        ],
    },
)
