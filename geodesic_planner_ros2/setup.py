from setuptools import find_packages, setup

package_name = 'geodesic_planner_ros2'

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
    maintainer='samubuntu',
    maintainer_email='Troyandme04@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'geodesic_planner_server = geodesic_planner_ros2.planner_server:main',
            'geodesic_planner_client = geodesic_planner_ros2.planner_client:main'
        ],
    },
)
