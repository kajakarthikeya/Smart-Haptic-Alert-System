"""
Command-Line Interface for Real-Time Sound Recognition.

Usage:
  python -m app.ai.inference.cli --list-devices
  python -m app.ai.inference.cli --test-file dataset/processed/car_horn/sample_0.wav
  python -m app.ai.inference.cli --live [--device-id 1] [--threshold 0.70]
"""

import argparse
import sys
import time
from typing import Optional

from app.ai.inference.audio_capture import AudioDeviceManager
from app.ai.inference.prediction import PredictionResult, PredictionStatus
from app.ai.inference.realtime_recognizer import RealtimeSoundRecognizer
from app.utils.logger import get_logger

logger = get_logger(__name__)


def format_header() -> str:
    return (
        "=" * 65 + "\n"
        "           SMART HAPTIC ALERT SYSTEM\n"
        "           Real-Time Sound Recognition\n"
        + "=" * 65
    )


def print_prediction_banner(res: PredictionResult) -> None:
    """Renders a structured, clean terminal card for a recognition event."""
    status_tag = "[CONFIRMED]" if res.status == PredictionStatus.CONFIRMED else ("[TENTATIVE]" if res.status == PredictionStatus.TENTATIVE else "[LOW_CONFIDENCE]")
    print("\n" + "-" * 65)
    print(f"Timestamp        : {res.timestamp}")
    print(f"Detected Sound   : {status_tag} {res.predicted_class}")
    print(f"Confidence       : {res.confidence * 100:.1f}%")
    print(f"Status           : {res.status.value}")
    print(f"Latency          : {res.latency.total_ms:.1f} ms (prep={res.latency.preprocessing_ms:.1f}ms, feat={res.latency.feature_extraction_ms:.1f}ms, infer={res.latency.inference_ms:.1f}ms)")
    
    # Top probability distribution
    probs_str = " | ".join([f"{k}: {v*100:.1f}%" for k, v in sorted(res.probabilities.items(), key=lambda x: -x[1])[:3]])
    print(f"Top Probabilities: {probs_str}")
    print("-" * 65)


def run_test_file_mode(file_path: str, threshold: float) -> int:
    """Executes single test file recognition and prints output card."""
    print(format_header())
    print(f"\nEvaluating test audio file: {file_path}")
    print(f"Configured Confidence Threshold: {threshold * 100:.1f}%\n")

    recognizer = RealtimeSoundRecognizer(confidence_threshold=threshold)
    try:
        result = recognizer.recognize_file(file_path)
        print_prediction_banner(result)
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Recognition failed for file '{file_path}': {exc}", file=sys.stderr)
        return 1


def run_live_streaming_mode(device_id: Optional[int], threshold: float, hop_sec: float) -> int:
    """Starts live interactive microphone stream with graceful Ctrl+C interrupt."""
    print(format_header())
    print("\nInitializing real-time acoustic recognizer...")

    try:
        recognizer = RealtimeSoundRecognizer(confidence_threshold=threshold)
        dev_info = AudioDeviceManager.get_device_info(device_id)
        print(f"Input Microphone : [{dev_info.device_id}] {dev_info.name}")
        print(f"Sampling Rate    : {recognizer.feature_pipeline.sample_rate} Hz (Mono)")
        print(f"Window Duration  : {recognizer.feature_pipeline.duration_sec:.1f} seconds (88,200 samples)")
        print(f"Hop Interval     : {hop_sec:.2f} seconds")
        print(f"Confidence Gate  : {threshold * 100:.1f}%")
        print(f"Stability Buffer : {recognizer.stabilizer.buffer_size} windows (agreement={recognizer.stabilizer.required_agreement})")
        print("\nListening for environmental sounds... (Press Ctrl+C to stop)")

        def _on_prediction(res: PredictionResult):
            print_prediction_banner(res)

        recognizer.start_streaming(callback=_on_prediction, hop_duration_sec=hop_sec)

        while True:
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 65)
        print("Stopping real-time sound recognition session...")
        recognizer.stop_streaming()

        summary = recognizer.get_session_summary()
        print("\nSESSION PERFORMANCE SUMMARY")
        print("-" * 65)
        print(f"Total Windows Processed     : {summary['total_windows_processed']}")
        print(f"Confident Predictions       : {summary['confident_predictions']}")
        print(f"Low-Confidence Windows      : {summary['low_confidence_predictions']}")
        print(f"Confirmed Alerts            : {summary['confirmed_alerts']}")
        print(f"Tentative Alerts            : {summary['tentative_alerts']}")
        print(f"Average Preprocessing Time  : {summary['average_preprocessing_ms']:.2f} ms")
        print(f"Average Feature Extr. Time  : {summary['average_feature_extraction_ms']:.2f} ms")
        print(f"Average Model Inference Time: {summary['average_inference_ms']:.2f} ms")
        print(f"Average Total Latency       : {summary['average_total_latency_ms']:.2f} ms")
        print("=" * 65)
        print("Session terminated cleanly.")
        return 0

    except Exception as exc:
        print(f"\n[FATAL ERROR] Live streaming encountered an exception: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smart Haptic Alert System - Real-Time Sound Recognition CLI",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input recording hardware devices.",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Path to an offline WAV audio file for verification without microphone.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live real-time continuous microphone recognition loop.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="Input microphone device ID (defaults to host default).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Confidence threshold [0.0 - 1.0] (default: 0.70).",
    )
    parser.add_argument(
        "--hop-sec",
        type=float,
        default=1.0,
        help="Window hop interval in seconds (default: 1.0).",
    )

    args = parser.parse_args()

    if args.list_devices:
        devices = AudioDeviceManager.list_input_devices()
        print("\nAvailable Audio Input Hardware Devices:")
        print("-" * 75)
        print(f"{'ID':<4} | {'Default':<8} | {'Max In':<8} | {'Sample Rate':<12} | {'Device Name'}")
        print("-" * 75)
        for d in devices:
            def_mark = "YES" if d.is_default else "NO"
            print(f"{d.device_id:<4} | {def_mark:<8} | {d.max_input_channels:<8} | {int(d.default_samplerate):<12} | {d.name}")
        print("-" * 75)
        return 0

    if args.test_file:
        return run_test_file_mode(args.test_file, args.threshold)

    # If --live is specified or no other action given
    return run_live_streaming_mode(args.device_id, args.threshold, args.hop_sec)


if __name__ == "__main__":
    sys.exit(main())
