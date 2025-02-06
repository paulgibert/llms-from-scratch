import torch
from llama3.positional_embeddings import RotaryPositionalEmbeddings


def test_RotaryPositionalEmbeddings():
    """
    Test that RotaryPostionalEmbeddings outputs the correct values and shape.
    """
    x = torch.tensor([[1, 2, 3, 4, 5, 6],
                         [3, 6, 9, 12, 15, 18]])
    x = x.unsqueeze(0).unsqueeze(2)
    
    expected = torch.tensor([[-2.234741690198506,
                              0.0770037537313969,
                              2.20646280601572,
                              4.486816453307315,
                              4.894944908921481,
                              6.086009722192667],
                             
                             [-3.8166975381605397,
                              -5.516594955423072,
                              5.337523535619122,
                              14.01823250294815,
                              14.525581921811277,
                              18.38497946239673]]).unsqueeze(0).unsqueeze(2)
    model = RotaryPositionalEmbeddings(6, 16, 1234)
    actual = model(x, 2)
    assert(actual.shape == expected.shape)
    assert(torch.allclose(actual, expected, atol=1e-6))
