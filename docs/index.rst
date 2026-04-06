rocm-trace-lite
===============

Self-contained GPU kernel profiler for ROCm. **Zero roctracer/rocprofiler-sdk dependency.**

Captures GPU kernel dispatch timestamps using only HSA runtime interception,
writing to SQLite in the standard `RPD <https://github.com/ROCm/rocmProfileData>`_ format.
One command to profile, one file to analyze.

.. code-block:: bash

   rpd-lite trace -o trace.db python3 my_model.py

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
       <h3>RPD Compatible</h3>
       <p>Outputs standard RPD SQLite schema. Works with existing RPD ecosystem tools and SQL queries.</p>
     </div>
   </div>


Quick Example
-------------

.. code-block:: bash

   # Install
   pip install rocm-trace-lite

   # Trace a workload
   rpd-lite trace -o trace.db python3 my_model.py

   # View top kernels
   rpd-lite summary trace.db

   # Open in Perfetto
   # trace.json.gz is auto-generated, open at https://ui.perfetto.dev

Sample output:

.. code-block:: text

   Trace: trace.db (728 GPU ops)

   Kernel                                              Calls  Total(us)  Avg(us)      %
   ========================================================================================
   Cijk_Ailk_Bljk_HHS_BH_MT128x128x128                  240    28252.9    117.7   21.8
   ncclDevKernel_Generic                                  160    29747.8    185.9   23.0
   __amd_rocclr_fillBufferAligned.kd                     7900    27929.8      3.5   21.6

   GPU Utilization:
     GPU 0: 0.13% (2630 ops, 17.2ms busy)
     GPU 1: 0.11% (2430 ops, 15.0ms busy)


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

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   contributing
   changelog
