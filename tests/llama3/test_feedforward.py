import torch
from llama3.feedforward import SWIGLUFeedForward


def test_SWIGLUFeedForward():
    """
    Check that SWIGLUFeedForward outputs the correct shape.
    """
    x = torch.rand(2, 16, 4)
    model = SWIGLUFeedForward(4, 8)
    output = model(x)
    assert(output.shape == (2, 16, 4))
