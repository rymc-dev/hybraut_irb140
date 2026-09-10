from glob import glob

from setuptools import find_packages, setup

package_name = 'hybraut_irb140'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # trained YOLO weights: lego_detector needs one dropped in before build;
        # ball_detector works without (falls back to pretrained yolov8n.pt) but
        # will use a fine-tuned perception/ball_perception/weights/*.pt if present.
        # the .pt files themselves are git-ignored.
        ('share/' + package_name + '/weights',
            glob('lego_detection/weights/*.pt')
            + glob('perception/ball_perception/weights/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ryan',
    maintainer_email='ryanmckee47@icloud.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hybraut_irb140 = hybraut_irb140.hybraut_irb140:main',
            'hybraut_irb140_line_detector = hybraut_irb140.line_detector:main',
            'hybraut_irb140_line_follower = hybraut_irb140.hybraut_irb140_line_follower:main',
            'hybraut_irb140_lego_detector = hybraut_irb140.lego_detector:main',
            'hybraut_irb140_ball_detector = hybraut_irb140.ball_detector:main',
        ],
    },
)
