from app.core.config import Settings, recommend_coach_runtime_profile, recommend_ollama_num_ctx


def test_recommend_ollama_num_ctx_cpu_amd64():
    assert recommend_ollama_num_ctx(
        use_gpu=False,
        gpu_vram_gb=0.0,
        hardware_target="amd64",
    ) == 4096


def test_recommend_ollama_num_ctx_cpu_arm64():
    assert recommend_ollama_num_ctx(
        use_gpu=False,
        gpu_vram_gb=0.0,
        hardware_target="arm64",
    ) == 2048


def test_recommend_ollama_num_ctx_gpu_tiers():
    assert recommend_ollama_num_ctx(
        use_gpu=True,
        gpu_vram_gb=8.0,
        hardware_target="amd64",
    ) == 8192
    assert recommend_ollama_num_ctx(
        use_gpu=True,
        gpu_vram_gb=12.0,
        hardware_target="amd64",
    ) == 16384
    assert recommend_ollama_num_ctx(
        use_gpu=True,
        gpu_vram_gb=24.0,
        hardware_target="amd64",
    ) == 24576
    assert recommend_ollama_num_ctx(
        use_gpu=True,
        gpu_vram_gb=32.0,
        hardware_target="amd64",
    ) == 32768


def test_settings_assigns_dynamic_ollama_num_ctx_when_unset():
    settings = Settings(
        USE_GPU=True,
        GPU_VRAM=12.0,
        HARDWARE_TARGET="amd64",
    )

    assert settings.OLLAMA_NUM_CTX == 16384


def test_settings_keeps_explicit_ollama_num_ctx_override():
    settings = Settings(
        USE_GPU=True,
        GPU_VRAM=24.0,
        HARDWARE_TARGET="amd64",
        OLLAMA_NUM_CTX=12288,
    )

    assert settings.OLLAMA_NUM_CTX == 12288


def test_recommend_coach_runtime_profile_cpu():
    assert recommend_coach_runtime_profile(
        use_gpu=False,
        gpu_vram_gb=0.0,
        high_vram_threshold_gb=16.0,
    ) == "cpu"


def test_recommend_coach_runtime_profile_gpu_tiers():
    assert recommend_coach_runtime_profile(
        use_gpu=True,
        gpu_vram_gb=4.0,
        high_vram_threshold_gb=16.0,
    ) == "cpu"
    assert recommend_coach_runtime_profile(
        use_gpu=True,
        gpu_vram_gb=8.0,
        high_vram_threshold_gb=16.0,
    ) == "gpu_constrained"
    assert recommend_coach_runtime_profile(
        use_gpu=True,
        gpu_vram_gb=24.0,
        high_vram_threshold_gb=16.0,
    ) == "gpu_full"
