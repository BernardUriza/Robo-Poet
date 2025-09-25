#!/usr/bin/env python3
"""
Test script for Django + robo_poet.py integration
"""

import requests
import json
import time
import sys

def test_training_api():
    """Test the training API endpoint."""
    print("=== Testing Training API ===")

    # Test data
    training_data = {
        'model_name': 'test_web_model',
        'cycles': 1,
        'epochs': 2,  # Short test
        'phase': '3'  # Intelligent cycle
    }

    try:
        # Test HTTP API
        print("1. Testing HTTP API...")
        response = requests.post(
            'http://localhost:8000/training/api/sessions/',
            json=training_data,
            timeout=5
        )

        print(f"Response status: {response.status_code}")
        print(f"Response data: {response.json()}")

        if response.status_code == 200:
            print("[OK] HTTP API working")
            session_id = response.json().get('session_id')
            return session_id
        else:
            print("[FAIL] HTTP API failed")
            return None

    except requests.exceptions.ConnectionError:
        print("[FAIL] Cannot connect to Django server. Make sure it's running on port 8000")
        return None
    except Exception as e:
        print(f"[FAIL] HTTP API error: {e}")
        return None

def test_gpu_status():
    """Test GPU status endpoint."""
    print("\n=== Testing GPU Status API ===")

    try:
        response = requests.get('http://localhost:8000/training/api/gpu-status/')
        print(f"GPU Status: {response.json()}")
        return True
    except Exception as e:
        print(f"GPU Status error: {e}")
        return False

def test_websocket():
    """Test WebSocket connection (basic test)."""
    print("\n=== Testing WebSocket Connection ===")

    try:
        import websocket

        def on_message(ws, message):
            print(f"WebSocket message: {message}")

        def on_error(ws, error):
            print(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")

        def on_open(ws):
            print("[OK] WebSocket connection established")
            # Send test message
            ws.send(json.dumps({'type': 'get_gpu_status'}))
            time.sleep(2)
            ws.close()

        ws = websocket.WebSocketApp(
            "ws://localhost:8000/ws/training/global/",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )

        # Run for 5 seconds
        ws.run_forever()
        return True

    except ImportError:
        print("[FAIL] websocket-client not installed")
        return False
    except Exception as e:
        print(f"[FAIL] WebSocket error: {e}")
        return False

def test_django_integration():
    """Test the complete Django integration."""
    print("=== Testing Django Integration ===")

    # Test robo_poet.py with Django environment
    import subprocess
    import os

    try:
        env = os.environ.copy()
        env['DJANGO_RUN'] = 'true'
        env['TRAINING_SESSION_ID'] = '999'  # Test session ID

        # Test a quick robo_poet.py execution
        process = subprocess.Popen(
            ['python', 'robo_poet.py', '--headless'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        # Send test input (Phase 3, test model, 1 cycle, 1 epoch)
        input_data = "3\ntest_model\n1\n1\n\n"
        try:
            stdout, stderr = process.communicate(input=input_data, timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            print("[FAIL] robo_poet.py timed out")
            return False

        print(f"robo_poet.py stdout: {stdout[:500]}...")
        print(f"robo_poet.py stderr: {stderr[:200]}...")

        if "[DJANGO]" in stdout:
            print("[OK] Django mode detected in robo_poet.py")
            return True
        else:
            print("[FAIL] Django mode not working")
            return False

    except subprocess.TimeoutExpired:
        print("[FAIL] robo_poet.py timed out")
        process.kill()
        return False
    except Exception as e:
        print(f"[FAIL] Django integration error: {e}")
        return False

def main():
    """Run all tests."""
    print("Starting Django + Robo-Poet Integration Tests")
    print("=" * 50)

    results = {
        'gpu_status': test_gpu_status(),
        'http_api': test_training_api() is not None,
        'websocket': test_websocket(),
        'django_integration': test_django_integration()
    }

    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    for test, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {test}: {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print(f"\nOverall: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("All tests passed! Integration is working correctly.")
        return 0
    else:
        print("Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())