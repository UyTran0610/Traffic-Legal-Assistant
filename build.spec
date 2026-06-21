# =============================================================
# build.spec — PyInstaller build file cho TroLyLuatGT
# Flet 0.80.5 — Python 3.12 — Windows 11
# =============================================================

block_cipher = None

import sys, os
site_pkg = next(p for p in sys.path if 'site-packages' in p and os.path.isdir(p))

def sp(rel):
    return os.path.join(site_pkg, rel)

a = Analysis(
    ['ui.py'],
    pathex=['.'],

    # ----------------------------------------------------------
    # BINARIES: C extension (.pyd) cần copy thủ công
    # ----------------------------------------------------------
    binaries=[
        # chromadb rust bindings — PyInstaller không tự tìm được
        (sp('chromadb_rust_bindings/chromadb_rust_bindings.pyd'), 'chromadb_rust_bindings'),

        # [PATCH] onnxruntime providers — PyInstaller hay bỏ sót các .dll này
        # nằm cùng thư mục với onnxruntime .pyd, cần copy đúng chỗ
        (sp('onnxruntime/capi/onnxruntime_providers_shared.dll'), 'onnxruntime/capi'),
    ],

    # ----------------------------------------------------------
    # DATAS: bundle toàn bộ thư viện có lazy import động
    # ----------------------------------------------------------
    datas=[
        # Flet 0.80.5
        (sp('flet_desktop/app'),      'flet_desktop/app'),
        (sp('flet'),                  'flet'),

        # AI / ML libs — dùng lazy import, PyInstaller hay bỏ sót
        (sp('transformers'),          'transformers'),
        (sp('chromadb'),              'chromadb'),
        (sp('chromadb_rust_bindings'),'chromadb_rust_bindings'),

        # LangChain
        (sp('langchain_community'),   'langchain_community'),
        (sp('langchain_chroma'),      'langchain_chroma'),
        (sp('langchain_openai'),      'langchain_openai'),
        (sp('langchain_core'),        'langchain_core'),

        # OpenAI client
        (sp('openai'),                'openai'),

        # [PATCH] tokenizers — thư viện Rust, có file .json schema bên trong
        # cần bundle cả thư mục để AutoTokenizer load đúng
        (sp('tokenizers'),            'tokenizers'),

        # [PATCH] httpx / certifi — cần cho ChatOpenAI gọi llama-server
        # certifi chứa bundle CA certificate (cacert.pem), thiếu → SSL error
        (sp('certifi'),               'certifi'),
    ],

    # ----------------------------------------------------------
    # HIDDEN IMPORTS
    # ----------------------------------------------------------
    hiddenimports=[
        # --- Flet ---
        'flet',
        'flet_desktop',
        'flet_desktop.app',
        'flet.app',
        'flet.fastapi',
        'flet.controls',
        'flet.utils',

        # --- ONNX Runtime ---
        'onnxruntime',
        'onnxruntime.capi',
        'onnxruntime.capi._pybind_state',
        'onnxruntime.capi.onnxruntime_inference_collection',

        # --- Transformers / Tokenizers ---
        'transformers',
        'transformers.models.auto',
        'transformers.models.auto.modeling_auto',
        'transformers.models.auto.configuration_auto',
        'transformers.models.auto.tokenization_auto',
        'transformers.models.bert',
        'transformers.models.roberta',
        'transformers.models.xlm_roberta',
        'tokenizers',
        'huggingface_hub',
        'huggingface_hub.file_download',

        # --- ChromaDB ---
        'chromadb',
        'chromadb.api',
        'chromadb.api.client',
        'chromadb.segment',
        'chromadb.segment.impl',
        'chromadb.segment.impl.manager',
        'chromadb.segment.impl.manager.local',
        'chromadb.telemetry',
        'chromadb.telemetry.product',
        'chromadb.telemetry.product.posthog',
        'chromadb.utils',
        'chromadb.utils.embedding_functions',
        'chromadb_rust_bindings',
        'chromadb_rust_bindings.chromadb_rust_bindings',

        # --- LangChain ---
        'langchain_core',
        'langchain_core.prompts',
        'langchain_core.output_parsers',
        'langchain_core.embeddings',
        'langchain_community',
        'langchain_community.document_loaders',
        'langchain_community.document_loaders.pdf',
        'langchain_chroma',
        'langchain_openai',
        'langchain_text_splitters',

        # --- PDF ---
        'pymupdf',
        'fitz',

        # --- Numpy ---
        'numpy',
        'numpy.core',
        'numpy.core._multiarray_umath',

        # --- HTTP / Async ---
        'httpx',
        'httpcore',
        'anyio',
        'anyio._backends._asyncio',
        'openai',
        'openai.resources',

        # --- Stdlib ---
        'sqlite3',
        'uuid',
        'asyncio',
        'asyncio.events',
        'concurrent.futures',
        'threading',
        'queue',

        # [PATCH] ctypes — dùng bởi Windows Job Object trong patch llama-server
        # Các module con này hay bị miss khi console=False
        'ctypes',
        'ctypes.wintypes',
        'ctypes.util',

        # [PATCH] signal — dùng cho SIGINT/SIGTERM handler trong patch
        'signal',

        # [PATCH] certifi — SSL cert bundle cho httpx/ChatOpenAI
        'certifi',

        # [PATCH] urllib.request — dùng cho ping health-check & watchdog
        # thường có nhưng khai báo rõ để chắc chắn
        'urllib.request',
        'urllib.error',
        'urllib.parse',
    ],

    excludes=[
        'torch',
        'torchvision',
        'torchaudio',
        'tensorflow',
        'keras',
        'jax',
        'optimum',
        'IPython',
        'jupyter',
        'notebook',
        'matplotlib',
        'PIL',
        'Pillow',
        'scipy',
        'sklearn',
        'pandas',
        'pyarrow',
        'cv2',
        'tkinter',
        '_tkinter',
        'wx',
        'PyQt5',
        'PyQt6',
    ],

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TroLyLuatGT',
    debug=False,
    bootloader_ignore_signals=False,  # [GHI CHÚ] Giữ False — True sẽ chặn SIGINT/SIGTERM
                                      # của patch không chạy được
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='TroLyLuatGT',
)
