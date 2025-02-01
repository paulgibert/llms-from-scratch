import torch
from llama3.positional_embeddings import RotaryPositionalEmbeddings


def test_RotaryPositionalEmbeddings():
    """
    Test that RotaryPostionalEmbeddings outputs the correct values and shape.
    """
    model = RotaryPositionalEmbeddings(6, 16, 1234)
    head = torch.tensor([[1, 2, 3, 4, 5, 6]])
    x = head.unsqueeze(0).unsqueeze(0).repeat(2,10,2,1)
    
    expected = torch.tensor([[0.024311087352848097,
                             0.8094645731931012,
                             2.1507162639364763,
                             2.89468708210845,
                             4.101866406367904,
                             4.975647839431939,
                             6.024240163457471]]).unsqueeze(0).unsqueeze(0).repeat(2,10,2,1)
    actual = model(x, 1)
    assert(actual.shape == expected.shape)
    assert(actual==expected)
