from setuptools import setup

setup(
    name='isaac_ros_bridge',
    version='0.0.1',
    packages=['isaac_ros_bridge',
              'isaac_ros_bridge.planner',
              'isaac_ros_bridge.models',
              'isaac_ros_bridge.utils',
              'isaac_ros_bridge.ros'],
    package_dir={'': 'src'},
    install_requires=[],
)
