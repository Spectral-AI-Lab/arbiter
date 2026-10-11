"""Ensure GLiNER loads the pinned snapshot rather than a moving repository."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmark.gliner_adapter import GLiNERDecisionAgent
from benchmark.run import MODELS


def test_pinned_snapshot_is_loaded():
    extractor = Mock()
    repo, revision = MODELS['gliner25-decide']
    assert len(revision) == 40
    with patch.dict(sys.modules, {'gliner2': SimpleNamespace(AutoExtractor=extractor)}):
        with patch('huggingface_hub.snapshot_download', return_value='/pinned/snapshot') as download:
            GLiNERDecisionAgent(repo, 'cpu', revision=revision)
    download.assert_called_once_with(repo_id=repo, revision=revision)
    extractor.from_pretrained.assert_called_once_with('/pinned/snapshot', map_location='cpu')
