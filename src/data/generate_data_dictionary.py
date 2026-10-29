import pandas as pd

df = pd.read_parquet('data/raw/agent_diagnostics/signals.parquet')

descriptions = {
    'task_id': 'Unique benchmark problem / task identifier',
    'model': 'Underlying LLM backbone evaluated (e.g. claude-haiku-4-5, claude-sonnet-4-6)',
    'agent_name': 'Autonomous agent system framework (claude-code, openhands, cursor-cli)',
    'config_name': 'Specific agent configuration profile applied during execution',
    'benchmark': 'Evaluation benchmark suite (e.g. openhands, sdlc_fix, org_migration)',
    'reward': 'Continuous evaluation score awarded by benchmark verifier (0.0 to 1.0)',
    'passed': 'Primary ground-truth binary outcome: True if all tests passed, False if failed',
    'has_verifier_result': 'Indicator whether benchmark sandbox verifier executed to completion',
    'total_turns': 'Total user/agent conversational interaction turns in the trial',
    'tool_calls_total': 'Total count of external tool/command invocations by the agent',
    'search_tool_calls': 'Count of text or keyword search tool invocations (Grep, Find)',
    'edit_tool_calls': 'Count of code editing tool invocations (StrReplace, Write)',
    'code_nav_tool_calls': 'Count of AST/symbol code navigation invocations (GoToDef, FindRefs)',
    'semantic_search_tool_calls': 'Count of vector/embedding semantic search invocations',
    'unique_files_read': 'Number of distinct repository files inspected by the agent',
    'unique_files_edited': 'Number of distinct repository files modified by the agent',
    'files_read_list': 'List of absolute file paths inspected during the session',
    'files_edited_list': 'List of absolute file paths modified during the session',
    'error_count': 'Total count of environment, syntax, or execution errors returned to agent',
    'retry_count': 'Total count of retry actions triggered after failed commands',
    'trajectory_length': 'Sequential length of recorded tool execution steps',
    'has_result_json': 'Whether trial execution metadata result JSON was preserved',
    'has_trajectory': 'Whether step-level trajectory JSON was logged',
    'duration_seconds': 'Total wall-clock runtime duration of the agent session in seconds',
    'rate_limited': 'Boolean flag indicating whether execution was throttled by API rate limits',
    'exception_crashed': 'Boolean flag indicating whether the agent runtime suffered an uncaught crash',
    'patch_size_lines': 'Total number of modified lines in the final generated patch/diff',
    'tool_call_sequence': 'Ordered sequential list of tools invoked throughout the execution trajectory',
    'benchmark_source': 'Source provenance type of the benchmark trial',
    'trial_path': 'Internal filesystem path of the original raw trial directory',
    'trial_id': 'Unique 32-character hexadecimal trial session identifier',
    'trial_id_full': 'Full 64-character SHA identifier of the trial session'
}

data_types = {
    'task_id': 'Categorical / ID',
    'model': 'Categorical',
    'agent_name': 'Categorical',
    'config_name': 'Categorical',
    'benchmark': 'Categorical',
    'reward': 'Continuous Numerical [0, 1]',
    'passed': 'Binary Target (Boolean)',
    'has_verifier_result': 'Binary Indicator',
    'total_turns': 'Discrete Numerical',
    'tool_calls_total': 'Discrete Numerical',
    'search_tool_calls': 'Discrete Numerical',
    'edit_tool_calls': 'Discrete Numerical',
    'code_nav_tool_calls': 'Discrete Numerical',
    'semantic_search_tool_calls': 'Discrete Numerical',
    'unique_files_read': 'Discrete Numerical',
    'unique_files_edited': 'Discrete Numerical',
    'files_read_list': 'Structured Sequence (List)',
    'files_edited_list': 'Structured Sequence (List)',
    'error_count': 'Discrete Numerical',
    'retry_count': 'Discrete Numerical',
    'trajectory_length': 'Discrete Numerical',
    'has_result_json': 'Binary Indicator',
    'has_trajectory': 'Binary Indicator',
    'duration_seconds': 'Continuous Numerical (Seconds)',
    'rate_limited': 'Binary Indicator',
    'exception_crashed': 'Binary Indicator',
    'patch_size_lines': 'Discrete Numerical',
    'tool_call_sequence': 'Sequential Trajectory (Array)',
    'benchmark_source': 'Categorical',
    'trial_path': 'Path / Metadata',
    'trial_id': 'Unique Key (Grouping Key)',
    'trial_id_full': 'Unique Hash'
}

rows = []
for col in df.columns:
    rows.append({
        'field_name': col,
        'data_type': data_types.get(col, str(df[col].dtype)),
        'null_count': int(df[col].isnull().sum()),
        'null_pct': f"{(df[col].isnull().sum() / len(df) * 100):.2f}%",
        'original_vs_derived': 'Original Raw Feature',
        'description': descriptions.get(col, '')
    })

dict_df = pd.DataFrame(rows)
dict_df.to_csv('DATA_DICTIONARY.csv', index=False)
print(f'Successfully generated DATA_DICTIONARY.csv with {len(dict_df)} fields.')
