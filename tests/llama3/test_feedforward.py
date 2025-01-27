import torch
from llama3.feedforward import SWIGLUFeedForward



def test_SWIGLUFeedForward():
    """
    Check that SWIGLUFeedForward outputs the correct shape.
    """
    model = SWIGLUFeedForward(4, 8)
    x = torch.rand(2, 16, 4)
    output = model(x)
    assert(output.shape == (2, 16, 4))
