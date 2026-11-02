def pytest_configure(config):
    config.addinivalue_line("markers", "tutorial: runs a documentation tutorial script (slow)")
