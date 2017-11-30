import pytest

from pydhamed.util.testing import data


def test_data(data):
    with pytest.raises(RuntimeError):
        data['foo']
