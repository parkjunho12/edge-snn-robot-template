from setuptools import setup
import os
from glob import glob

package_name = 'edge_snn_robot'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Junho Park',
    maintainer_email='ghdlwnsgh25@gmail.com',
    description='Edge SNN Robot Control - EMG to Robot Hand/Arm with Fake Hardware',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'emg_intent_node = edge_snn_robot.emg_intent_node:main',
            'servo_cmd_node = edge_snn_robot.servo_cmd_node:main',
            'fake_hardware_node = edge_snn_robot.fake_hardware:main',
        ],
    },
)