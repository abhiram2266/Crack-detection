"""Application service container (built once per API process)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from auth.credentials import get_credential
from auth.security import load_secret
from memory.cloud_store_interface import CloudStoreInterface, HttpCloudStore
from memory.store import LocalStore
from memory.sync import SyncManager
from ml.context_risk import ContextRiskService
from ml.condition_classifier import ConditionClassifier, load_condition_classifier
from ml.config import cfg
from ml.detector import Detector, build_detector
from ml.embedding_model import AssetEmbeddingModel
from ml.escalation import EscalationService
from ml.mission import MissionService
from ml.semantic_memory import SemanticMemoryEngine

from api.ws import Hub

logger = logging.getLogger(__name__)


@dataclass
class Services:
    config: Dict[str, Any]
    store: LocalStore
    embedder: AssetEmbeddingModel
    classifier: ConditionClassifier
    detector: Detector
    engine: SemanticMemoryEngine
    sync: SyncManager
    escalation: EscalationService
    mission: MissionService
    hub: Hub
    jwt_secret: str
    live: Any = None
    context: Optional[ContextRiskService] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def build_cloud(config: Dict[str, Any]) -> Optional[CloudStoreInterface]:
    c = cfg(config, "cloud")
    if not c["enabled"]:
        return None
    afile = cfg(config, "auth", "credentials_file")
    user = os.environ.get(c["username_env"]) or get_credential(c["username_env"], afile)
    password = os.environ.get(c["password_env"]) or get_credential(c["password_env"], afile)
    if not user or not password:
        logger.warning("Cloud sync disabled: %s / %s not set (run scripts/seed_users.py)", c["username_env"], c["password_env"])
        return None
    return HttpCloudStore(c["base_url"], user, password)


def build_services(
    config: Dict[str, Any],
    *,
    store: Optional[LocalStore] = None,
    embedder: Optional[AssetEmbeddingModel] = None,
    classifier: Optional[ConditionClassifier] = None,
    detector: Optional[Detector] = None,
    cloud: Optional[CloudStoreInterface] = None,
    use_cloud: bool = True,
) -> Services:
    store = store or LocalStore(config)
    site = cfg(config, "sites")
    store.ensure_site(site["default_site_id"], site["default_site_name"])
    embedder = embedder or AssetEmbeddingModel.from_config(config)
    classifier = classifier or load_condition_classifier(config, embedder)
    detector = detector or build_detector(config, embedder)
    engine = SemanticMemoryEngine(config, store, embedder, classifier, detector)
    context = ContextRiskService(config, store)
    engine.context = context
    if cloud is None and use_cloud:
        cloud = build_cloud(config)
    sync = SyncManager(config, store, cloud)
    hub = Hub(int(cfg(config, "live", "ws_queue_size")))
    escalation = EscalationService(config, store, publisher=hub.publish_to_user)
    engine.add_listener(escalation.on_event)
    mission = MissionService(store, engine, sync, escalation, context)
    a = cfg(config, "auth")
    secret = load_secret(a["secret_env"], a["secret_file"])
    return Services(config, store, embedder, classifier, detector, engine, sync, escalation, mission, hub, secret, context=context)
