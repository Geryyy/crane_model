from glob import glob

from setuptools import setup

PACKAGE = "crane_model"

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=[PACKAGE],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml"]),
        (
            f"share/{PACKAGE}/config",
            glob("config/*.yaml") + glob("config/*.json") + glob("config/*.srdf"),
        ),
        (f"share/{PACKAGE}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE}/description", glob("test/description/*.urdf")),
    ],
    install_requires=["setuptools"],
    # colcon picks its pytest step off `tests_require`, not off package.xml, so
    # without this line `test/` is never run.
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Architecture maintainers",
    maintainer_email="maintainers@example.invalid",
    description="Python crane model API: symbolic and numeric boundaries for planning and control.",
    license="Apache-2.0",
)
