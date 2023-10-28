import torch
from classify import classify_binary_sentiment


def test_classify_sentiment():
    logits = torch.FloatTensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            [1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
        ]
    )

    assert classify_binary_sentiment(logits, torch.LongTensor([1, 2])) == [1, 0, 1]
    assert classify_binary_sentiment(logits, torch.LongTensor([2, 1])) == [0, 1, 0]

    assert classify_binary_sentiment(logits, torch.LongTensor([0, 4])) == [1, 0, 1]
    assert classify_binary_sentiment(logits, torch.LongTensor([0, 3])) == [1, 0, 0]
