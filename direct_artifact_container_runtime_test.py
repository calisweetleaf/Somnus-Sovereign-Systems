#!/usr/bin/env python3
"""
Direct function test script for artifact_container_runtime.py
No pytest, just manual calls and print statements.
"""
import sys
import traceback
from uuid import uuid4
from datetime import datetime, timezone

try:
    from backend.artifacts.artifact_container_runtime import (
        ContainerState, ContainerConfig, ContainerMetrics, ContainerExecutionContext,
        UnlimitedContainerRuntime
    )
except Exception as e:
    print("Import error:", e)
    sys.exit(1)

# Helper to print test results

def print_result(name, result):
    print(f"[TEST] {name}: {result}")

# Test ContainerState enum
try:
    print_result("ContainerState.CREATING", ContainerState.CREATING)
    print_result("ContainerState.RUNNING", ContainerState.RUNNING)
except Exception as e:
    print("ContainerState test failed:", e)

# Test ContainerConfig
try:
    config = ContainerConfig()
    print_result("Default ContainerConfig.image_name", config.image_name)
    print_result("Default ContainerConfig.privileged", config.privileged)
    custom_config = ContainerConfig(image_name="custom:image", privileged=False)
    print_result("Custom ContainerConfig.image_name", custom_config.image_name)
    print_result("Custom ContainerConfig.privileged", custom_config.privileged)
except Exception as e:
    print("ContainerConfig test failed:", e)

# Test ContainerMetrics
try:
    metrics = ContainerMetrics()
    print_result("Default ContainerMetrics.cpu_percent", metrics.cpu_percent)
    print_result("Default ContainerMetrics.memory_usage_mb", metrics.memory_usage_mb)
except Exception as e:
    print("ContainerMetrics test failed:", e)

# Test ContainerExecutionContext
try:
    artifact_id = uuid4()
    context = ContainerExecutionContext(
        artifact_id=artifact_id,
        container_id="test_id",
        container_name="test_container",
        config=config,
        state=ContainerState.RUNNING,
        created_at=datetime.now(timezone.utc)
    )
    print_result("ContainerExecutionContext.artifact_id", context.artifact_id)
    print_result("ContainerExecutionContext.state", context.state)
except Exception as e:
    print("ContainerExecutionContext test failed:", e)

# Test UnlimitedContainerRuntime instantiation
try:
    # Use None for managers, minimal config
    runtime = UnlimitedContainerRuntime(
        artifact_manager=None,
        vm_manager=None,
        memory_manager=None,
        config={}
    )
    print_result("UnlimitedContainerRuntime instantiated", True)
except Exception as e:
    print("UnlimitedContainerRuntime instantiation failed:", e)
    traceback.print_exc()

# Note: Most runtime methods require Docker and full system integration.
# For direct function testing, we avoid actual Docker calls.

print("\nDirect function tests complete.")
