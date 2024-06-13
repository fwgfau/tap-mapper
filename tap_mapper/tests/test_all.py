import unittest
import HtmlTestRunner


def discover():
    loader = unittest.TestLoader()
    tests = loader.discover(".")
    test_runner = HtmlTestRunner.HTMLTestRunner(output="../../public/")
    test_runner.run(tests)


if __name__ == "__main__":
    discover()
