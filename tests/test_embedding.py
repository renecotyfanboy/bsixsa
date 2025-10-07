import pytest
import scipy as sp
import numpy as np
from pytest import fixture
from bsixsa.embedding import MultipleEmbedding
from bsixsa.embedding.summary import GlobalSummaryEmbedding, GlobalSummaryConfig, LocalSumEmbedding, LocalSumAndRatioEmbedding, basic_global_stats
from tests.conftest import mock_data

custom_global_stats = {
    "kurtosis": lambda x: sp.stats.kurtosis(x, axis=1),
    "skewness": lambda x: sp.stats.skew(x, axis=1),
    "entropy": lambda x: sp.stats.entropy(x, axis=1)
}

@pytest.mark.parametrize("config", [GlobalSummaryConfig(basic_global_stats), GlobalSummaryConfig(custom_global_stats)], ids=["standard", "custom"])
def test_global_summary_embedding(mock_data, config):

    embedding = GlobalSummaryEmbedding(config)
    embedded = embedding(mock_data)

    assert embedded.shape == (len(mock_data), len(embedding.names)), "Embedding shape is not correct"

    embedded_single = embedding(mock_data[0, :])

    assert embedded_single.shape == (1, len(embedding.names)), "Embedding on single array should add a dimension"


def test_merge_summaries(mock_data):

    embedding_1 = GlobalSummaryEmbedding(GlobalSummaryConfig(basic_global_stats))
    embedding_2 = GlobalSummaryEmbedding(GlobalSummaryConfig(custom_global_stats))
    embedding_3 = MultipleEmbedding([embedding_1, embedding_2])

    assert (name in embedding_3.names for name in embedding_1.names) and (name in embedding_3.names for name in embedding_2.names), "Embedding names not in merged"

    embedded_1 = embedding_1(mock_data)
    embedded_2 = embedding_2(mock_data)
    embedded_3 = embedding_3(mock_data)

    assert embedded_3.shape == (len(mock_data), embedded_1.shape[1] + embedded_2.shape[1]), "Embedding shape is not correct"

@pytest.mark.parametrize("energy_grid", [np.linspace(2., 8., 11), np.geomspace(5, 10, 51)], ids=["Small linear", "Large log"])
def test_local_sum_embedding(mock_data, energy_grid):

    embedding = LocalSumEmbedding(energy_grid)
    embedded = embedding(mock_data)

    assert embedded.shape == (len(mock_data), len(energy_grid)-1), "Embedding shape is not correct"
    assert len(embedding.names) == len(energy_grid) - 1, "Embedding names is not correct"

@pytest.mark.parametrize("energy_grid", [np.linspace(2., 8., 11), np.geomspace(5, 10, 51)], ids=["Small linear", "Large log"])
def test_local_sum_and_ratio_embedding(mock_data, energy_grid):

    embedding = LocalSumAndRatioEmbedding(energy_grid)
    embedded = embedding(mock_data)

    assert embedded.shape == (len(mock_data), 2*len(energy_grid)-3), "Embedding shape is not correct"
    assert len(embedding.names) == 2*len(energy_grid)-3, "Embedding names is not correct"