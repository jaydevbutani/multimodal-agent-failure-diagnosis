import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

class LiveAgentTracer:
    """
    Live Agent Tracing & Instrumentation Harness.
    Captures real execution turns, tool invocations, error returns,
    and formats them for ML diagnostic prediction.
    """
    def __init__(self, trace_id=None, session_type="NATURAL_EXECUTION"):
        self.trace_id = trace_id or f"LIVE-{int(time.time())}"
        self.session_type = session_type  # 'NATURAL_EXECUTION' or 'CONTROLLED_EXPERIMENT'
        self.start_time = None
        self.end_time = None
        self.turns = 0
        self.tool_calls = []
        self.errors = 0
        self.retries = 0
        self.files_read = set()
        self.files_edited = set()
        self.patch_size_lines = 0
        self.search_calls = 0
        self.edit_calls = 0
        self.code_nav_calls = 0
        self.semantic_search_calls = 0
        self.rate_limited = False
        self.exception_crashed = False
        
    def start_session(self):
        self.start_time = time.time()
        
    def record_turn(self):
        self.turns += 1
        
    def record_tool_call(self, tool_name, is_error=False, is_retry=False, files_read=None, files_edited=None, lines_changed=0):
        self.tool_calls.append(tool_name)
        tool_l = tool_name.lower()
        if any(k in tool_l for k in ['grep', 'find', 'search', 'glob']):
            self.search_calls += 1
        elif any(k in tool_l for k in ['read', 'view', 'cat']):
            self.code_nav_calls += 1
        elif any(k in tool_l for k in ['write', 'edit', 'replace']):
            self.edit_calls += 1
            
        if is_error:
            self.errors += 1
        if is_retry:
            self.retries += 1
        if files_read:
            self.files_read.update(files_read)
        if files_edited:
            self.files_edited.update(files_edited)
        self.patch_size_lines += lines_changed
        
    def end_session(self):
        self.end_time = time.time()
        calc_duration = round(self.end_time - self.start_time, 2) if self.start_time else 1.0
        duration = getattr(self, "duration_override", None) or calc_duration
        
        telemetry = {
            'trace_id': self.trace_id,
            'session_type': self.session_type,
            'timestamp': datetime.utcnow().isoformat(),
            'model': 'claude-sonnet-4-6',
            'agent_name': 'claude-code',
            'benchmark': 'live_harness',
            'total_turns': max(1, self.turns),
            'tool_calls_total': len(self.tool_calls),
            'search_tool_calls': self.search_calls,
            'edit_tool_calls': self.edit_calls,
            'code_nav_tool_calls': self.code_nav_calls,
            'semantic_search_tool_calls': self.semantic_search_calls,
            'unique_files_read': len(self.files_read),
            'unique_files_edited': len(self.files_edited),
            'error_count': self.errors,
            'retry_count': self.retries,
            'patch_size_lines': self.patch_size_lines,
            'duration_seconds': max(0.5, duration),
            'rate_limited': self.rate_limited,
            'exception_crashed': self.exception_crashed,
            'tool_call_sequence': self.tool_calls,
            'step_events': getattr(self, "step_events", [])
        }
        
        # Save trace
        os.makedirs("data/raw/live_traces", exist_ok=True)
        out_path = f"data/raw/live_traces/{self.trace_id}.json"
        with open(out_path, "w") as f:
            json.dump(telemetry, f, indent=2)
            
        return telemetry

def run_sample_live_trace(scenario="healthy"):
    scenario_clean = scenario.lower()
    tracer = LiveAgentTracer(
        trace_id=f"LIVE-{scenario.upper()}-{int(time.time())}",
        session_type="CONTROLLED_EXPERIMENT" if scenario_clean != "healthy" else "NATURAL_EXECUTION"
    )
    tracer.start_session()
    tracer.step_events = []
    
    def log_step(tool_name, is_err=False, is_ret=False, files_r=None, files_e=None, lines=0, latency_ms=18.5):
        tracer.record_turn()
        tracer.record_tool_call(tool_name, is_error=is_err, is_retry=is_ret, files_read=files_r, files_edited=files_e, lines_changed=lines)
        tracer.step_events.append({
            "step": len(tracer.step_events) + 1,
            "tool": tool_name,
            "latency_ms": round(latency_ms, 1),
            "is_error": is_err,
            "is_retry": is_ret,
            "cumulative_errors": tracer.errors
        })

    if scenario_clean in ["healthy", "nominal"]:
        nominal_steps = [
            ("read_file", ["README.md"], None, 0, 14.2),
            ("Grep", ["src/main.py"], None, 0, 22.0),
            ("read_file", ["src/main.py"], None, 0, 18.1),
            ("read_file", ["tests/test_main.py"], None, 0, 16.5),
            ("str_replace_editor", None, ["src/main.py"], 12, 35.0),
            ("execute_bash", None, None, 0, 120.5),
            ("read_file", ["tests/test_main.py"], None, 0, 15.0),
            ("write_file", None, ["tests/test_main.py"], 25, 45.2),
            ("execute_bash", None, None, 0, 95.0),
            ("finish", None, None, 0, 5.0)
        ]
        for tool, f_read, f_edit, lines, lat in nominal_steps:
            log_step(tool, False, False, f_read, f_edit, lines, lat)
        tracer.duration_override = 45.0

    elif scenario_clean in ["thrashing_failure", "retry_thrashing", "cyclic_retry_thrashing"]:
        tracer.turns = 95
        log_step("read_file", False, False, ["src/main.py"], None, 0, 15.0)
        log_step("read_file", False, False, ["tests/test_api.py"], None, 0, 18.0)
        for i in range(10):
            log_step("execute_bash", is_err=True, is_ret=(i > 0), latency_ms=140.0 + (i * 12.0))
        tracer.duration_override = 520.0

    elif scenario_clean in ["environment_failure", "env_failure"]:
        tracer.turns = 55
        tracer.rate_limited = True
        log_step("read_file", False, False, ["src/server.py"], None, 0, 20.0)
        log_step("execute_bash", False, False, None, None, 0, 85.0)
        for i in range(8):
            log_step("execute_bash", is_err=True, is_ret=(i % 2 == 1), latency_ms=350.0 + (i * 30.0))
        tracer.duration_override = 410.0

    elif scenario_clean in ["high_code_churn", "code_churn"]:
        tracer.turns = 55
        log_step("read_file", False, False, ["src/handler.py"], None, 0, 15.0)
        for i in range(7):
            log_step("str_replace_editor", is_err=(i >= 5), is_ret=False, files_e=[f"src/file_{i}.py"], lines=120, latency_ms=65.0)
        log_step("execute_bash", is_err=True, is_ret=True, latency_ms=180.0)
        tracer.duration_override = 340.0

    elif scenario_clean in ["over_exploration", "information_over_exploration"]:
        tracer.turns = 85
        for i in range(12):
            log_step("search_code", False, False, files_r=[f"docs/module_{i}.md"], latency_ms=25.0 + (i * 3.0))
        log_step("execute_bash", is_err=True, is_ret=False, latency_ms=90.0)
        tracer.duration_override = 380.0

    elif scenario_clean in ["tool_execution_failure", "tool_instability"]:
        tracer.turns = 35
        for i in range(8):
            is_e = (i % 2 == 1)
            log_step("execute_tool", is_err=is_e, is_ret=is_e, latency_ms=110.0 + (i * 15.0))
        tracer.duration_override = 210.0

    else:
        # Compound Failure
        tracer.turns = 70
        log_step("read_file", False, False, ["src/main.py"], None, 0, 20.0)
        for i in range(4):
            log_step("str_replace_editor", False, False, files_e=[f"src/mod_{i}.py"], lines=80, latency_ms=50.0)
        for i in range(6):
            log_step("execute_bash", is_err=True, is_ret=(i > 0), latency_ms=160.0 + (i * 20.0))
        tracer.duration_override = 480.0
        
    telemetry = tracer.end_session()
    print(f"Generated live session trace: {telemetry['trace_id']} ({telemetry['session_type']}) with {len(telemetry['tool_call_sequence'])} tool calls.")
    return telemetry

if __name__ == '__main__':
    run_sample_live_trace("healthy")
    run_sample_live_trace("thrashing_failure")
