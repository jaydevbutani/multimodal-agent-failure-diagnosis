import pandas as pd

features = [
    # Original Features
    {
        'Feature': 'total_turns',
        'Original_or_Derived': 'Original',
        'Formula_or_Source': 'Raw count of dialogue interaction turns',
        'Reason': 'Measures conversational depth and task complexity',
        'Leakage_Risk': 'None (Computed within trial session)'
    },
    {
        'Feature': 'tool_calls_total',
        'Original_or_Derived': 'Original',
        'Formula_or_Source': 'Raw sum of tool invocations',
        'Reason': 'Captures overall agent activity level',
        'Leakage_Risk': 'None'
    },
    {
        'Feature': 'search_tool_calls',
        'Original_or_Derived': 'Original',
        'Formula_or_Source': 'Count of search operations (Grep, Find)',
        'Reason': 'Reflects information-gathering phase intensity',
        'Leakage_Risk': 'None'
    },
    {
        'Feature': 'edit_tool_calls',
        'Original_or_Derived': 'Original',
        'Formula_or_Source': 'Count of edit operations (Write, Replace)',
        'Reason': 'Reflects solution implementation intensity',
        'Leakage_Risk': 'None'
    },
    {
        'Feature': 'code_nav_tool_calls',
        'Original_or_Derived': 'Original',
        'Formula_or_Source': 'Count of AST navigation calls (Def, Refs)',
        'Reason': 'Reflects structured code exploration',
        'Leakage_Risk': 'None'
    },
    {
        'Feature': 'unique_files_read',
        'Original_or_Derived': 'Original',
        'Formula_or_Source': 'Distinct file count read',
        'Reason': 'Context breadth indicator',
        'Leakage_Risk': 'None'
    },
    {
        'Feature': 'unique_files_edited',
        'Original_or_Derived': 'Original',
        'Formula_or_Source': 'Distinct file count edited',
        'Reason': 'Blast radius indicator of changes',
        'Leakage_Risk': 'None'
    },
    {
        'Feature': 'error_count',
        'Original_or_Derived': 'Original',
        'Formula_or_Source': 'Total error responses returned by environment',
        'Reason': 'Direct execution friction indicator',
        'Leakage_Risk': 'None'
    },
    {
        'Feature': 'retry_count',
        'Original_or_Derived': 'Original',
        'Formula_or_Source': 'Total repeated commands after failure',
        'Reason': 'Diagnostic thrashing / loop indicator',
        'Leakage_Risk': 'None'
    },
    {
        'Feature': 'duration_seconds',
        'Original_or_Derived': 'Original',
        'Formula_or_Source': 'Wall-clock session execution duration',
        'Reason': 'Temporal resource consumption',
        'Leakage_Risk': 'None'
    },
    {
        'Feature': 'patch_size_lines',
        'Original_or_Derived': 'Original',
        'Formula_or_Source': 'Modified line count in final git patch',
        'Reason': 'Code modification volume',
        'Leakage_Risk': 'None'
    },
    {
        'Feature': 'rate_limited',
        'Original_or_Derived': 'Original',
        'Formula_or_Source': 'API rate limit throttle boolean',
        'Reason': 'Environment disruption signal',
        'Leakage_Risk': 'None'
    },
    {
        'Feature': 'exception_crashed',
        'Original_or_Derived': 'Original',
        'Formula_or_Source': 'Runtime unhandled crash boolean',
        'Reason': 'Fatal infrastructure failure indicator',
        'Leakage_Risk': 'None'
    },

    # Derived Features
    {
        'Feature': 'DERIVED_error_rate',
        'Original_or_Derived': 'DERIVED FEATURE',
        'Formula_or_Source': 'error_count / max(1, tool_calls_total)',
        'Reason': 'Normalizes error counts by agent activity level',
        'Leakage_Risk': 'Zero: calculated on individual trial telemetry'
    },
    {
        'Feature': 'DERIVED_retry_ratio',
        'Original_or_Derived': 'DERIVED FEATURE',
        'Formula_or_Source': 'retry_count / max(1, error_count)',
        'Reason': 'Measures agent propensity to enter repetitive retry loops upon encountering errors',
        'Leakage_Risk': 'Zero: strictly within-sample arithmetic'
    },
    {
        'Feature': 'DERIVED_search_to_edit_ratio',
        'Original_or_Derived': 'DERIVED FEATURE',
        'Formula_or_Source': 'search_tool_calls / max(1, edit_tool_calls)',
        'Reason': 'Differentiates over-exploration/churn from focused editing',
        'Leakage_Risk': 'Zero: strictly within-sample arithmetic'
    },
    {
        'Feature': 'DERIVED_edit_churn',
        'Original_or_Derived': 'DERIVED FEATURE',
        'Formula_or_Source': 'patch_size_lines / max(1, unique_files_edited)',
        'Reason': 'Average code lines modified per file; high values signal uncontrolled rewrites',
        'Leakage_Risk': 'Zero: strictly within-sample arithmetic'
    },
    {
        'Feature': 'DERIVED_tool_intensity',
        'Original_or_Derived': 'DERIVED FEATURE',
        'Formula_or_Source': 'tool_calls_total / max(1, total_turns)',
        'Reason': 'Frequency of tool usage per turn; distinguishes conversational vs tool-heavy steps',
        'Leakage_Risk': 'Zero: strictly within-sample arithmetic'
    },
    {
        'Feature': 'DERIVED_read_to_edit_ratio',
        'Original_or_Derived': 'DERIVED FEATURE',
        'Formula_or_Source': 'unique_files_read / max(1, unique_files_edited)',
        'Reason': 'Investigative breadth relative to execution modification',
        'Leakage_Risk': 'Zero: strictly within-sample arithmetic'
    },
    {
        'Feature': 'DERIVED_time_per_turn',
        'Original_or_Derived': 'DERIVED FEATURE',
        'Formula_or_Source': 'duration_seconds / max(1, total_turns)',
        'Reason': 'Latency pace per conversational turn; flags network stall or complex deliberation',
        'Leakage_Risk': 'Zero: strictly within-sample arithmetic'
    },
    {
        'Feature': 'DERIVED_tool_entropy',
        'Original_or_Derived': 'DERIVED FEATURE',
        'Formula_or_Source': '-sum(p * log(p)) over tool call frequency distribution in tool_call_sequence',
        'Reason': 'Information-theoretic diversity of tool usage; low entropy signals single-tool looping',
        'Leakage_Risk': 'Zero: computed solely on tool_call_sequence within trial'
    },
    {
        'Feature': 'DERIVED_log_duration',
        'Original_or_Derived': 'DERIVED FEATURE',
        'Formula_or_Source': 'log1p(duration_seconds)',
        'Reason': 'Mitigates extreme right-skewness of session execution time',
        'Leakage_Risk': 'Zero: monotonic elementwise transformation'
    },
    {
        'Feature': 'DERIVED_log_patch_size',
        'Original_or_Derived': 'DERIVED FEATURE',
        'Formula_or_Source': 'log1p(patch_size_lines)',
        'Reason': 'Stabilizes variance across patch sizes from 1 line to 10,000+ lines',
        'Leakage_Risk': 'Zero: monotonic elementwise transformation'
    }
]

df_feat = pd.DataFrame(features)
df_feat.to_csv('FEATURE_DICTIONARY.csv', index=False)
print(f'Successfully generated FEATURE_DICTIONARY.csv with {len(df_feat)} documented features.')
