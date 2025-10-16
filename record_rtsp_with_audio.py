#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime


def build_ffmpeg_cmd(rtsp_url: str,
                     output_path: str,
                     duration: int | None,
                     segment_time: int | None,
                     tcp_transport: bool,
                     map_all: bool,
                     map_video: str | None,
                     map_audio: str | None,
                     hwaccel: str | None,
                     container_format: str,
                     audio_codec: str,
                     extra_input_args: list[str],
                     extra_output_args: list[str]) -> list[str]:
    cmd: list[str] = ["ffmpeg", "-hide_banner", "-y"]

    if tcp_transport:
        cmd += ["-rtsp_transport", "tcp"]

    if hwaccel:
        cmd += ["-hwaccel", hwaccel]

    # Input options: low-latency, allowed to drop late frames
    cmd += extra_input_args
    cmd += ["-i", rtsp_url]

    # Map streams
    if map_all:
        cmd += ["-map", "0", "-map", "-0:d"]  # everything except data streams
    else:
        if map_video:
            cmd += ["-map", map_video]
        else:
            cmd += ["-map", "0:v:0?"]
        if map_audio:
            cmd += ["-map", map_audio]
        else:
            cmd += ["-map", "0:a:0?"]

    # Video: copy to avoid re-encode (assuming H.264)
    cmd += ["-c:v", "copy"]

    # Audio codec handling
    if audio_codec == "copy":
        cmd += ["-c:a", "copy"]
    elif audio_codec == "pcm_mulaw":
        cmd += ["-c:a", "pcm_mulaw", "-ar", "8000", "-ac", "1"]
    elif audio_codec == "pcm_alaw":
        cmd += ["-c:a", "pcm_alaw", "-ar", "8000", "-ac", "1"]
    else:  # aac default
        cmd += ["-c:a", "aac", "-b:a", "128k", "-ac", "1", "-ar", "48000"]

    # Container/muxer
    if container_format == "mp4":
        cmd += ["-f", "mp4", "-movflags", "+faststart"]
    elif container_format == "mkv":
        cmd += ["-f", "matroska"]
    elif container_format == "ts":
        cmd += ["-f", "mpegts"]
    elif container_format == "mov":
        cmd += ["-f", "mov"]

    # Duration
    if duration and duration > 0 and not segment_time:
        cmd += ["-t", str(duration)]

    # Segmented recording
    if segment_time and segment_time > 0:
        # Switch to segment muxer; output_path should include %03d pattern
        cmd += ["-f", "segment", "-segment_time", str(segment_time), "-reset_timestamps", "1"]

    cmd += extra_output_args
    cmd += [output_path]
    return cmd


def ensure_output_path(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)


def default_output_path(base_dir: str, prefix: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{prefix}_{ts}.mp4")


def main() -> int:
    parser = argparse.ArgumentParser(description="Record RTSP video+audio from PTZ camera using ffmpeg")
    parser.add_argument("--url", required=True, help="RTSP URL including credentials, e.g. rtsp://user:pass@ip:554/Stream/Live/101")
    parser.add_argument("--out", default=None, help="Output file path (.mp4). If using segmenting, include %03d in filename")
    parser.add_argument("--out-dir", default="recordings", help="Directory for output if --out not provided")
    parser.add_argument("--prefix", default="ptz_recording", help="Filename prefix when auto-generating output path")
    parser.add_argument("--duration", type=int, default=0, help="Max duration in seconds (0 = unlimited)")
    parser.add_argument("--segment", type=int, default=0, help="Segment duration in seconds (0 = disabled)")
    parser.add_argument("--tcp", action="store_true", help="Force RTSP over TCP (recommended)")
    parser.add_argument("--map-all", action="store_true", help="Map all streams and drop data tracks")
    parser.add_argument("--video-map", default=None, help="ffmpeg -map spec for video (e.g., 0:v:0)")
    parser.add_argument("--audio-map", default=None, help="ffmpeg -map spec for audio (e.g., 0:a:0)")
    parser.add_argument("--hwaccel", default=None, help="ffmpeg hwaccel (e.g., cuda, vaapi). Only affects decoding if re-encode.")
    parser.add_argument("--format", default="mp4", choices=["mp4", "mkv", "mov", "ts"], help="Container format")
    parser.add_argument("--audio-codec", default="aac", choices=["aac", "copy", "pcm_mulaw", "pcm_alaw"], help="Audio codec handling")
    parser.add_argument("--print-streams", action="store_true", help="Probe input streams and exit")
    parser.add_argument("--extra-in", nargs="*", default=["-stimeout", "5000000", "-use_wallclock_as_timestamps", "1", "-fflags", "+genpts"], help="Extra ffmpeg input args")
    parser.add_argument("--extra-out", nargs="*", default=[], help="Extra ffmpeg output args")

    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found in PATH. Please install ffmpeg.", file=sys.stderr)
        return 1

    if args.print_streams:
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            print("Error: ffprobe not found in PATH. Please install ffmpeg/ffprobe.", file=sys.stderr)
            return 1
        probe_cmd = [ffprobe, "-hide_banner", "-v", "error", "-show_streams", "-of", "json"]
        if args.tcp:
            probe_cmd += ["-rtsp_transport", "tcp"]
        probe_cmd += [args.url]
        print("Probing streams:")
        print(" ".join(subprocess.list2cmdline([x]) for x in probe_cmd))
        return subprocess.call(probe_cmd)

    # Decide output path
    if args.out:
        output_path = args.out
    else:
        if args.segment and args.segment > 0:
            # segmented output pattern
            ext = "mp4" if args.format == "mp4" else ("mkv" if args.format == "mkv" else args.format)
            output_path = os.path.join(args.out_dir, f"{args.prefix}_%03d.{ext}")
        else:
            ext = "mp4" if args.format == "mp4" else ("mkv" if args.format == "mkv" else args.format)
            output_path = os.path.join(args.out_dir, f"{args.prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}")

    ensure_output_path(output_path)

    # Warn about incompatible combos
    if args.format == "mp4" and args.audio_codec in ("copy", "pcm_mulaw", "pcm_alaw"):
        print("Note: MP4 typically requires AAC/MP3 for audio. Consider --audio-codec aac or --format mkv.", file=sys.stderr)

    cmd = build_ffmpeg_cmd(
        rtsp_url=args.url,
        output_path=output_path,
        duration=args.duration if args.duration > 0 else None,
        segment_time=args.segment if args.segment > 0 else None,
        tcp_transport=args.tcp,
        map_all=args.map_all,
        map_video=args.video_map,
        map_audio=args.audio_map,
        hwaccel=args.hwaccel,
        container_format=args.format,
        audio_codec=args.audio_codec,
        extra_input_args=args.extra_in,
        extra_output_args=args.extra_out,
    )

    print("Running:")
    print(" ".join(subprocess.list2cmdline([x]) for x in cmd))

    try:
        proc = subprocess.Popen(cmd)
        proc.wait()
        return proc.returncode
    except KeyboardInterrupt:
        print("Interrupted. Stopping ffmpeg…")
        try:
            proc.terminate()
        except Exception:
            pass
        return 130


if __name__ == "__main__":
    raise SystemExit(main())


