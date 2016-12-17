import pytest
from keras import backend as K
from resnet import ResnetBuilder


DIM_ORDERING = {'th', 'tf'}


def test_resnet18():
    for ordering in DIM_ORDERING:
        K.set_image_dim_ordering(ordering)
        model = ResnetBuilder.build_resnet_18((3, 224, 224), 100)
        model.compile(loss="categorical_crossentropy", optimizer="sgd")
        assert True, "Failed to build with '{}' dim ordering".format(ordering)


def test_resnet34():
    for ordering in DIM_ORDERING:
        K.set_image_dim_ordering(ordering)
        model = ResnetBuilder.build_resnet_34((3, 224, 224), 100)
        model.compile(loss="categorical_crossentropy", optimizer="sgd")
        assert True, "Failed to build with '{}' dim ordering".format(ordering)


def test_resnet50():
    for ordering in DIM_ORDERING:
        K.set_image_dim_ordering(ordering)
        model = ResnetBuilder.build_resnet_50((3, 224, 224), 100)
        model.compile(loss="categorical_crossentropy", optimizer="sgd")
        assert True, "Failed to build with '{}' dim ordering".format(ordering)


def test_resnet101():
    for ordering in DIM_ORDERING:
        K.set_image_dim_ordering(ordering)
        model = ResnetBuilder.build_resnet_101((3, 224, 224), 100)
        model.compile(loss="categorical_crossentropy", optimizer="sgd")
        assert True, "Failed to build with '{}' dim ordering".format(ordering)


def test_resnet152():
    for ordering in DIM_ORDERING:
        K.set_image_dim_ordering(ordering)
        model = ResnetBuilder.build_resnet_152((3, 224, 224), 100)
        model.compile(loss="categorical_crossentropy", optimizer="sgd")
        assert True, "Failed to build with '{}' dim ordering".format(ordering)


if __name__ == '__main__':
    pytest.main([__file__])
