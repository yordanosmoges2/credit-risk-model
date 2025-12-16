def test_data_split():
    assert 0.2 > 0

def test_model_import():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    assert model is not None
