"""Independent graph/parametric reduction and explicit structural certificates."""
from .config import TDAConfig

__all__ = ["DeepTDA", "TDAConfig", "GraphEmbedding", "PrecomputedGraphEmbedding",
           "TranslationTangentDissimilarity"]
__version__ = "0.3.0"


def __getattr__(name):
    # Graph/structural users need not import Torch. Optional graph JIT and ANN
    # dependencies are still isolated in standalone worker processes.
    if name == "DeepTDA":
        from .estimator import DeepTDA
        globals()[name] = DeepTDA
        return DeepTDA
    if name == "GraphEmbedding":
        from .graph_embedding import GraphEmbedding
        globals()[name] = GraphEmbedding
        return GraphEmbedding
    if name == "PrecomputedGraphEmbedding":
        from .precomputed_graph import PrecomputedGraphEmbedding
        globals()[name] = PrecomputedGraphEmbedding
        return PrecomputedGraphEmbedding
    if name == "TranslationTangentDissimilarity":
        from .image_dissimilarity import TranslationTangentDissimilarity
        globals()[name] = TranslationTangentDissimilarity
        return TranslationTangentDissimilarity
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
