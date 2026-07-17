# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import time
from typing import Optional

import requests

from relax.backends.sglang.sglang_engine import SGLangEngine
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class SGLangOmniEngine(SGLangEngine):
    """External HTTP proxy for one SGLang-Omni router."""

    supports_elastic_scale = False

    def init(
        self,
        dist_init_addr: str,
        port: int,
        nccl_port: int | None,
        host: str | None = None,
        disaggregation_bootstrap_port: int | None = None,
        router_ip: str | None = None,
        router_port: int | None = None,
        init_external_kwargs: Optional[dict] = None,
        skip_dcs_registration: bool = False,
        skip_router_registration: bool = False,
    ) -> None:
        del nccl_port
        del disaggregation_bootstrap_port
        del init_external_kwargs
        del skip_dcs_registration
        del skip_router_registration
        self._validate_mode()
        if self.worker_type != "regular":
            raise ValueError("SGLang-Omni only supports regular (non-PD) rollout workers")

        self.node_rank = 0
        self.server_host = host or str(dist_init_addr).rsplit(":", 1)[0]
        self.server_port = int(port)
        self.router_ip = router_ip
        self.router_port = router_port
        self._skip_router_registration = True
        self.checkpoint_engine_client = None
        self._wait_healthy()
        self._validate_server()

    def _validate_mode(self) -> None:
        if not getattr(self.args, "rollout_external", False):
            raise ValueError("SGLang-Omni rollout requires --rollout-external")
        if getattr(self.args, "fully_async", False):
            raise ValueError("SGLang-Omni rollout does not support --fully-async")
        if getattr(self.args, "offload_rollout", False):
            raise ValueError("SGLang-Omni rollout does not support rollout offload")
        external_addrs = getattr(self.args, "rollout_external_engine_addrs", None)
        if not external_addrs or len(external_addrs) != 1:
            raise ValueError(
                "SGLang-Omni rollout requires exactly one external Omni router address"
            )

    def _wait_healthy(self, timeout: float = 300.0) -> None:
        url = f"http://{self.server_host}:{self.server_port}/health"
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(1)
        raise TimeoutError(f"Timed out waiting for SGLang-Omni at {url}") from last_error

    def _validate_server(self) -> None:
        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/model_info",
            timeout=10,
        )
        response.raise_for_status()

    def health_generate(self, timeout: float = 5.0) -> bool:
        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/health",
            timeout=timeout,
        )
        response.raise_for_status()
        return True

    def flush_cache(self) -> dict | None:
        return self._make_request("flush_cache")

    def register_to_router(
        self,
        bootstrap_port: int | None = None,
        strict: bool = True,
    ) -> bool:
        del bootstrap_port
        del strict
        return True

    def unregister_from_router(self) -> bool:
        return True

    def shutdown(self) -> None:
        logger.info(
            "Detach external SGLang-Omni proxy at %s:%s",
            self.server_host,
            self.server_port,
        )

    def release_memory_occupation(self) -> None:
        raise NotImplementedError("SGLang-Omni rollout offload is not supported")

    def resume_memory_occupation(self, tags: list[str] | None = None) -> None:
        del tags
        raise NotImplementedError("SGLang-Omni rollout offload is not supported")
