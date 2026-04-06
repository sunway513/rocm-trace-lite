"""Shared fixtures for rpd_lite tests."""
import os
import sqlite3
import pytest

RPDB_LITE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Schema that rpd_lite creates (must match rpd_lite.cpp)
REQUIRED_TABLES = [
    "rocpd_string", "rocpd_op", "rocpd_api", "rocpd_api_ops",
    "rocpd_kernelapi", "rocpd_copyapi", "rocpd_metadata", "rocpd_monitor",
]
REQUIRED_VIEWS = ["top", "busy"]


@pytest.fixture
def tmp_rpd(tmp_path):
    """Return a path for a temporary .db file."""
    return str(tmp_path / "test.db")


@pytest.fixture
def empty_rpd(tmp_rpd):
    """Create an empty RPD database with the correct schema."""
    # Load rpd2trace to get the schema indirectly, or just create it manually
    conn = sqlite3.connect(tmp_rpd)
    conn.executescript(SCHEMA_SQL)
    conn.close()
    return tmp_rpd


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rocpd_string (id INTEGER PRIMARY KEY, string TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS rocpd_op (
    id INTEGER PRIMARY KEY,
    gpuId INTEGER,
    queueId INTEGER,
    sequenceId INTEGER,
    completionSignal TEXT,
    start INTEGER,
    end INTEGER,
    description_id INTEGER REFERENCES rocpd_string(id),
    opType_id INTEGER REFERENCES rocpd_string(id)
);
CREATE TABLE IF NOT EXISTS rocpd_api (
    id INTEGER PRIMARY KEY,
    pid INTEGER,
    tid INTEGER,
    start INTEGER,
    end INTEGER,
    apiName_id INTEGER REFERENCES rocpd_string(id),
    args_id INTEGER REFERENCES rocpd_string(id)
);
CREATE TABLE IF NOT EXISTS rocpd_api_ops (
    id INTEGER PRIMARY KEY,
    api_id INTEGER REFERENCES rocpd_api(id),
    op_id INTEGER REFERENCES rocpd_op(id)
);
CREATE TABLE IF NOT EXISTS rocpd_kernelapi (
    api_id INTEGER PRIMARY KEY REFERENCES rocpd_api(id),
    stream TEXT,
    gridX INTEGER, gridY INTEGER, gridZ INTEGER,
    workgroupX INTEGER, workgroupY INTEGER, workgroupZ INTEGER,
    groupSegmentSize INTEGER, privateSegmentSize INTEGER,
    kernelName_id INTEGER REFERENCES rocpd_string(id)
);
CREATE TABLE IF NOT EXISTS rocpd_copyapi (
    api_id INTEGER PRIMARY KEY REFERENCES rocpd_api(id),
    stream TEXT,
    size INTEGER,
    dst TEXT, src TEXT,
    kind INTEGER, sync INTEGER
);
CREATE TABLE IF NOT EXISTS rocpd_metadata (
    id INTEGER PRIMARY KEY,
    tag TEXT, value TEXT
);
CREATE TABLE IF NOT EXISTS rocpd_monitor (
    id INTEGER PRIMARY KEY,
    deviceType TEXT, deviceId INTEGER,
    monitorType TEXT,
    start INTEGER, end INTEGER,
    value TEXT
);

CREATE VIEW IF NOT EXISTS top AS
SELECT
    s.string AS Name,
    COUNT(*) AS Calls,
    SUM(o.end - o.start) AS TotalNs,
    ROUND(AVG(o.end - o.start), 1) AS AvgNs,
    MIN(o.end - o.start) AS MinNs,
    MAX(o.end - o.start) AS MaxNs,
    ROUND(100.0 * SUM(o.end - o.start) / (SELECT SUM(end - start) FROM rocpd_op WHERE end > start AND gpuId >= 0), 2) AS Pct
FROM rocpd_op o
JOIN rocpd_string s ON o.description_id = s.id
WHERE o.end > o.start AND o.gpuId >= 0
GROUP BY s.string
ORDER BY TotalNs DESC;

CREATE VIEW IF NOT EXISTS busy AS
SELECT
    gpuId,
    ROUND(100.0 * SUM(end - start) / (MAX(end) - MIN(start)), 2) AS utilization_pct,
    COUNT(*) AS ops,
    SUM(end - start) AS busy_ns,
    MAX(end) - MIN(start) AS wall_ns
FROM rocpd_op
WHERE end > start
GROUP BY gpuId;
"""


def populate_synthetic_trace(rpd_path, num_kernels=100, num_gpus=1):
    """Populate an RPD database with synthetic kernel data."""
    conn = sqlite3.connect(rpd_path)
    conn.executescript(SCHEMA_SQL)

    kernel_names = [
        "Cijk_GEMM_kernel.kd",
        "elementwise_add_kernel.kd",
        "reduce_mean_kernel.kd",
        "softmax_kernel.kd",
        "layernorm_kernel.kd",
    ]

    # Insert kernel name strings
    for name in kernel_names:
        conn.execute("INSERT OR IGNORE INTO rocpd_string(string) VALUES(?)", (name,))
    conn.execute("INSERT OR IGNORE INTO rocpd_string(string) VALUES(?)", ("KernelExecution",))

    # Insert ops
    base_ns = 1000000000000  # 1 trillion ns
    for i in range(num_kernels):
        name = kernel_names[i % len(kernel_names)]
        gpu_id = i % num_gpus
        start_ns = base_ns + i * 10000  # 10us apart
        duration_ns = 3000 + (i % 5) * 1000  # 3-7us
        end_ns = start_ns + duration_ns

        conn.execute(
            "INSERT INTO rocpd_op(gpuId, queueId, sequenceId, start, end, description_id, opType_id) "
            "VALUES(?, ?, 0, ?, ?, "
            "(SELECT id FROM rocpd_string WHERE string=?), "
            "(SELECT id FROM rocpd_string WHERE string='KernelExecution'))",
            (gpu_id, gpu_id, start_ns, end_ns, name),
        )

    conn.commit()
    conn.close()
    return rpd_path


def _rocm_gpu_available():
    """Check if ROCm GPU is usable (not just /dev/kfd exists).

    Runs a subprocess to verify torch.cuda sees the GPU.
    Results are cached for the session.
    """
    if not hasattr(_rocm_gpu_available, "_cached"):
        import subprocess
        import sys
        try:
            r = subprocess.run(
                [sys.executable, "-c",
                 "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() > 0"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
            )
            _rocm_gpu_available._cached = (r.returncode == 0)
        except Exception:
            _rocm_gpu_available._cached = False
    return _rocm_gpu_available._cached


def _rocm_gpu_count():
    """Return number of GPUs visible to PyTorch, cached."""
    if not hasattr(_rocm_gpu_count, "_cached"):
        import subprocess
        import sys
        try:
            r = subprocess.run(
                [sys.executable, "-c",
                 "import torch; print(torch.cuda.device_count())"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, timeout=30
            )
            _rocm_gpu_count._cached = int(r.stdout.strip()) if r.returncode == 0 else 0
        except Exception:
            _rocm_gpu_count._cached = 0
    return _rocm_gpu_count._cached
