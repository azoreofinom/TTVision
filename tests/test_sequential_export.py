import importlib.util
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
# Export tests do not need the segmentation model or its ML dependencies.
with patch.dict(sys.modules, {'mask_processing': types.ModuleType('mask_processing')}):
    spec = importlib.util.spec_from_file_location('export_under_test', ROOT / 'src/edit_video.py')
    edit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(edit)


class SequentialExportTests(unittest.TestCase):
    def test_presets_cover_full_hardware_ranges_in_order(self):
        nvenc, amf = [], []
        for preset in edit.ENCODER_PRESETS:
            options = edit.encoder_options('h264_nvenc', preset)
            nvenc.append(options[options.index('-preset') + 1])
            options = edit.encoder_options('h264_amf', preset)
            amf.append(options[options.index('-quality') + 1])
            options = edit.encoder_options('libx264', preset)
            self.assertEqual(options[options.index('-preset') + 1], preset)
        self.assertEqual(list(dict.fromkeys(nvenc)), [f'p{i}' for i in range(1, 8)])
        self.assertEqual(list(dict.fromkeys(amf)), ['speed', 'balanced', 'quality'])
        self.assertEqual(nvenc, sorted(nvenc))
        self.assertEqual(amf, sorted(amf, key=['speed', 'balanced', 'quality'].index))

    def probe(self, path):
        result = subprocess.run([edit.media_tool('ffprobe'), '-v', 'error',
            '-show_streams', '-show_format', '-of', 'json', str(path)],
            capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def test_real_exports_with_and_without_audio(self):
        for audio in (False, True):
            with self.subTest(audio=audio), tempfile.TemporaryDirectory(dir=ROOT / 'build') as folder:
                work = Path(folder)
                source, output = work / 'source.mp4', work / 'result.mp4'
                cmd = [edit.media_tool('ffmpeg'), '-v', 'error', '-y', '-f', 'lavfi',
                    '-i', 'testsrc2=size=160x90:rate=30:duration=5']
                if audio:
                    cmd += ['-f', 'lavfi', '-i', 'sine=frequency=440:duration=5', '-c:a', 'aac']
                cmd += ['-c:v', 'libx264', '-preset', 'ultrafast', str(source)]
                subprocess.run(cmd, check=True)
                progress = []
                warnings = queue.Queue()
                self.assertTrue(edit.render_sequential_clips(str(source),
                    [(0.4, 1.4), (2.1, 3.1), (4, 4.5)], 'ultrafast', str(output), 30,
                    lambda n, d: progress.append(n / d), warning_queue=warnings, encoder="cpu"))
                info = self.probe(output)
                self.assertAlmostEqual(float(info['format']['duration']), 2.5, delta=0.15)
                self.assertEqual(any(s['codec_type'] == 'audio' for s in info['streams']), audio)
                subprocess.run([edit.media_tool('ffmpeg'), '-v', 'error', '-i', str(output),
                    '-f', 'null', '-'], check=True)
                self.assertEqual(progress, sorted(progress))
                self.assertLess(max(progress), 1)
                self.assertTrue(warnings.empty())
                self.assertEqual(sorted(p.name for p in work.iterdir()), ['result.mp4', 'source.mp4'])

    def test_failure_and_cancellation_preserve_destination_and_clean_up(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel), tempfile.TemporaryDirectory(dir=ROOT / 'build') as folder:
                output = Path(folder) / 'result.mp4'
                output.write_bytes(b'existing output')
                event = threading.Event()
                def fail(cmd, *args):
                    Path(cmd[-1]).write_bytes(b'partial clip')
                    if cancel:
                        event.set()
                    return False
                with patch.object(edit, 'has_audio_stream', return_value=False), patch.object(edit, 'run_ffmpeg_with_progress', side_effect=fail):
                    self.assertFalse(edit.render_sequential_clips('source.mp4', [(0, 1)],
                        'fast', str(output), 30, stop_event=event, encoder="cpu"))
                self.assertEqual(output.read_bytes(), b'existing output')
                self.assertEqual(list(Path(folder).iterdir()), [output])

    def test_probe_skips_unavailable_encoders(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as folder:
            seen = []
            def probe(cmd, *args):
                selected = cmd[cmd.index('-c:v') + 1]
                seen.append(selected)
                return selected == 'h264_amf'
            with patch.object(edit, 'run_ffmpeg_with_progress', side_effect=probe):
                self.assertEqual(edit.select_encoder('auto', 'fast', Path(folder), None), 'h264_amf')
            self.assertEqual(seen, ['h264_nvenc', 'h264_amf'])
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_hardware_failure_restarts_all_clips_on_cpu(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as folder:
            work = Path(folder)
            calls = []
            def render(cmd, *args):
                output = Path(cmd[-1])
                encoder = cmd[cmd.index('-c:v') + 1]
                calls.append((encoder, output.name))
                output.write_bytes(encoder.encode())
                if encoder == 'h264_nvenc' and output.name == 'clip_000001.mkv':
                    return False
                if encoder == 'copy':
                    self.assertEqual((output.parent / 'clip_000000.mkv').read_bytes(), b'libx264')
                    self.assertEqual((output.parent / 'clip_000001.mkv').read_bytes(), b'libx264')
                return True
            with patch.object(edit, 'has_audio_stream', return_value=False), patch.object(edit, 'select_encoder', return_value='h264_nvenc'), patch.object(edit, 'run_ffmpeg_with_progress', side_effect=render):
                self.assertTrue(edit.render_sequential_clips('input.mp4', [(0, 1), (2, 3)],
                    'fast', str(work / 'output.mp4'), 30))
            self.assertEqual(calls, [('h264_nvenc', 'clip_000000.mkv'), ('h264_nvenc', 'clip_000001.mkv'),
                ('libx264', 'clip_000000.mkv'), ('libx264', 'clip_000001.mkv'), ('copy', 'edited.mp4')])
            self.assertEqual([p.name for p in work.iterdir()], ['output.mp4'])

    def test_running_ffmpeg_can_be_cancelled(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as folder:
            output = Path(folder) / 'partial.mp4'
            event = threading.Event()
            timer = threading.Timer(0.5, event.set)
            timer.start()
            try:
                cmd = [edit.media_tool('ffmpeg'), '-y', '-re', '-f', 'lavfi', '-i',
                    'testsrc2=size=160x90:rate=30', '-t', '60', '-c:v', 'libx264', str(output)]
                self.assertFalse(edit.run_ffmpeg_with_progress(cmd, 60, None, 1800, 30, event))
                self.assertFalse(output.exists())
            finally:
                timer.cancel()

    def test_real_pipeline_cancellation_cleans_clips_and_join(self):
        for stage in ('clip', 'join'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory(dir=ROOT / 'build') as folder:
                work = Path(folder)
                source, output = work / 'source.mp4', work / 'result.mp4'
                subprocess.run([edit.media_tool('ffmpeg'), '-v', 'error', '-y',
                    '-f', 'lavfi', '-i', 'testsrc2=size=160x90:rate=30:duration=4',
                    '-c:v', 'libx264', '-preset', 'ultrafast', str(source)], check=True)
                output.write_bytes(b'existing output')
                event = threading.Event()
                observed = []
                original_runner = edit.run_ffmpeg_with_progress

                def run(cmd, duration, callback, frames, fps, stop, warnings):
                    is_join = 'concat' in cmd
                    if is_join == (stage == 'join'):
                        cmd = list(cmd)
                        cmd.insert(cmd.index('-i'), '-re')
                        def cancel_on_progress(n, total):
                            callback(n, total)
                            observed.extend(work.glob('ttvision-clips-*'))
                            event.set()
                        callback_to_use = cancel_on_progress
                    else:
                        callback_to_use = callback
                    return original_runner(cmd, duration, callback_to_use, frames, fps, stop, warnings)

                with patch.object(edit, 'run_ffmpeg_with_progress', side_effect=run):
                    self.assertFalse(edit.render_sequential_clips(str(source),
                        [(0, 1.5), (2, 3.5)], 'ultrafast', str(output), 30,
                        stop_event=event, encoder="cpu"))
                self.assertTrue(observed, 'Cancellation must occur after temporary media exists')
                self.assertTrue(event.is_set())
                self.assertEqual(output.read_bytes(), b'existing output')
                self.assertEqual(sorted(p.name for p in work.iterdir()), ['result.mp4', 'source.mp4'])


if __name__ == '__main__':
    unittest.main()
