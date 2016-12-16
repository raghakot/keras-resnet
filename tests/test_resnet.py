import pytest
from resnet import ResnetBuilder


def test_resnet18():
    model = ResnetBuilder.build_resnet_18((3, 224, 224), 100)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    assert True


def test_resnet34():
    model = ResnetBuilder.build_resnet_34((3, 224, 224), 100)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    assert True


def test_resnet50():
    model = ResnetBuilder.build_resnet_50((3, 224, 224), 100)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    assert True


def test_resnet101():
    model = ResnetBuilder.build_resnet_101((3, 224, 224), 100)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    assert True


def test_resnet152():
    model = ResnetBuilder.build_resnet_152((3, 224, 224), 100)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    assert True


if __name__ == '__main__':
    pytest.main([__file__])
