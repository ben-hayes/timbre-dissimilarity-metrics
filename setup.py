from distutils.core import setup

setup(
    name="timbremetrics",
    version="0.1",
    description="A suite of metrics for testing timbre dissimilarity models implemented with the TorchMetrics API",
    author="Ben Hayes",
    author_email="b.j.hayes@qmul.ac.uk",
    include_package_data=True,
    url="https://github.com/ben-hayes/timbre-dissimilarity-metrics",
    packages=["timbremetrics"],
)
