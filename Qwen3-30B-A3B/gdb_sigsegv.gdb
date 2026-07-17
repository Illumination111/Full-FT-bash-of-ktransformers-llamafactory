# Batch GDB commands for the Full-FT expert-base-only runner.
# Paths and core generation are controlled by:
#   FFT_GDB_LOG
#   FFT_GDB_CORE
#   FFT_GDB_GENERATE_CORE=0|1

set pagination off
set height 0
set width 0
set confirm off
set print pretty on
set print thread-events off
set breakpoint pending on

python
import datetime
import os
import gdb

sigsegv_seen = False


def _execute(command):
    try:
        gdb.execute(command)
    except gdb.error as exc:
        gdb.write(f"[gdb-capture] {command!r} failed: {exc}\n")


def _capture_sigsegv(event):
    global sigsegv_seen
    if not isinstance(event, gdb.SignalEvent) or event.stop_signal != "SIGSEGV":
        return

    sigsegv_seen = True
    log_path = os.environ.get("FFT_GDB_LOG", "gdb_sigsegv.log")
    core_path = os.environ.get("FFT_GDB_CORE", "core.sigsegv")
    generate_core = os.environ.get("FFT_GDB_GENERATE_CORE", "0") == "1"
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    gdb.execute(f"set logging file {log_path}")
    gdb.execute("set logging overwrite on")
    gdb.execute("set logging redirect off")
    gdb.execute("set logging enabled on")

    gdb.write("\n=== SIGSEGV captured by GDB ===\n")
    gdb.write(f"UTC time: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
    gdb.write(f"inferior PID: {gdb.selected_inferior().pid}\n")
    gdb.write(f"core requested: {generate_core}\n")

    gdb.write("\n--- Program and crashing thread ---\n")
    _execute("info program")
    _execute("info threads")
    _execute("bt full")
    _execute("info registers")
    _execute("x/16i $pc-32")
    _execute("info symbol $pc")

    gdb.write("\n--- All threads (up to 64 frames each) ---\n")
    _execute("thread apply all bt 64")

    gdb.write("\n--- Native library mappings ---\n")
    _execute("info sharedlibrary")
    _execute("info proc mappings")

    if generate_core:
        gdb.write(f"\n--- Generating optional core: {core_path} ---\n")
        _execute(f"generate-core-file {core_path}")
    else:
        gdb.write("\nCore generation disabled; use --gdb-core only when enough disk space is available.\n")

    gdb.write("=== End SIGSEGV capture ===\n")
    gdb.execute("set logging enabled off")


gdb.events.stop.connect(_capture_sigsegv)
end

# Stop before delivery so the event handler can inspect intact fault state.
handle SIGSEGV stop print nopass
run

# Preserve failure semantics for accelerate after the diagnostic capture.
python
if sigsegv_seen:
    gdb.execute("signal SIGSEGV")
end
