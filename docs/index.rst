rocm-trace-lite
===============

Self-contained GPU kernel profiler for ROCm. **Zero roctracer/rocprofiler-sdk dependency.**

Captures GPU kernel dispatch timestamps using only HSA runtime interception,
writing to a standard SQLite .db file.
One command to profile, one file to analyze.

.. code-block:: bash

   rtl trace -o trace.db python3 my_model.py

.. raw:: html

   <div class="feature-grid">
     <div class="feature-card">
       <h3>Lightweight</h3>
       <p>Single .so library. Only depends on libhsa-runtime64 + libsqlite3. No roctracer, no rocprofiler-sdk.</p>
     </div>
     <div class="feature-card">
       <h3>Multi-GPU Ready</h3>
       <p>Validated TP=8 on MI355X. Per-process trace files with automatic merge. Symmetric kernel capture across all ranks.</p>
     </div>
     <div class="feature-card">
       <h3>Perfetto Integration</h3>
       <p>Auto-generates compressed Perfetto JSON. Open in <a href="https://ui.perfetto.dev">ui.perfetto.dev</a> for timeline visualization.</p>
     </div>
     <div class="feature-card">
       <h3>SQLite Output</h3>
       <p>Outputs standard SQLite .db files. Compatible with <a href="https://github.com/ROCm/rocmProfileData">RPD</a> ecosystem tools and SQL queries.</p>
     </div>
   </div>


Quick Example
-------------

.. code-block:: bash

   # Install
   pip install rocm-trace-lite

   # Trace a workload
   rtl trace -o trace.db python3 my_model.py

   # View top kernels
   rtl summary trace.db

   # Open in Perfetto
   # trace.json.gz is auto-generated, open at https://ui.perfetto.dev

Sample output (DeepSeek-R1 671B, TP=8, MI355X):

.. code-block:: text

   Trace: trace.db (200590 GPU ops)

   Kernel                                             Calls  Total(ms)  Avg(us)     %
   ====================================================================================
   ncclDevKernel_Generic_1                             4851    7879.5   1624.5   55.2%
   aiter::fmoe_bf16_blockscaleFp8 (novs_silu)         3538    1239.8    350.4    8.7%
   aiter::reduce_scatter_cross_device_store<bf16,8>    8906     927.2    104.1    6.5%
   ck::kernel_gemm_xdl_cshuffle_v3 (blockscale)      20963     733.9     35.0    5.1%

   GPU Utilization:
     GPU 0: 51.2% (25074 ops, 7.3s busy)
     GPU 1: 50.8% (25081 ops, 7.2s busy)

**< 1% overhead** validated on 6 ATOM dashboard models (DeepSeek-R1, GPT-OSS, Kimi-K2.5, MiniMax-M2.5).
See `tutorial: profiling prefill vs decode <tutorial_roctx.html>`_ with built-in roctx markers.


Sample Results (MI355X, Apr 2026)
----------------------------------

Live benchmark results and kernel traces from the ATOM dashboard validation sweep:

.. raw:: html

   <div style="display:flex;gap:14px;margin:16px 0;">
     <a href="samples/index.html" style="display:inline-flex;align-items:center;gap:8px;background:#0d1220;border:1px solid #2a3a5a;border-radius:4px;padding:10px 18px;color:#29d9ff;text-decoration:none;font-family:monospace;font-size:13px;">
       📊 Overhead Benchmark Results
       <span style="font-size:11px;color:#6a7a9a">→ v0.3.3 · 8 models · MI355X</span>
     </a>
     <a href="samples/hot_trace.html" style="display:inline-flex;align-items:center;gap:8px;background:#0d1220;border:1px solid #2a3a5a;border-radius:4px;padding:10px 18px;color:#ff5c6a;text-decoration:none;font-family:monospace;font-size:13px;">
       ⚡ Hot Trace Viewer
       <span style="font-size:11px;color:#6a7a9a">→ Prefill vs Decode kernel hotspots</span>
     </a>
   </div>


Supported Hardware
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Architecture
     - GPU
     - Status
   * - CDNA 3 (gfx942)
     - MI300A, MI300X
     - Tested
   * - CDNA 3.5 (gfx950)
     - MI355X
     - Tested (TP=8 validated)
   * - CDNA 4 (gfx1250)
     - MI450
     - Tested (single GPU)
   * - CDNA 2 (gfx90a)
     - MI210, MI250, MI250X
     - Expected to work (untested)


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quickstart
   cli_reference

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorial_roctx

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   multi_gpu
   perfetto
   output_format

.. toctree::
   :maxdepth: 2
   :caption: Architecture
   :hidden:

   how_it_works
   comparison

Acknowledgments
----------------

This project was inspired by and builds upon the work of:

- **Jeff Daily**'s `ROCm Tracer for GPU (RTG) <https://github.com/ROCm/rtg_tracer>`_ — pioneered the HSA_TOOLS_LIB interception approach for lightweight GPU kernel tracing
- **Michael Wootton**'s `rocmProfileData (RPD) <https://github.com/ROCm/rocmProfileData>`_ — established the SQLite-based trace format and ecosystem tools that rocm-trace-lite is compatible with


.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   contributing
   changelog
