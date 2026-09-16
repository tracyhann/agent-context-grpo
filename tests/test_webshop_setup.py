#!/usr/bin/env python3
"""Provisioning failure guards; CPU-only, standard library, no network."""
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import setup_webshop as setup


class ProvisioningGuards(unittest.TestCase):
    def test_existing_bad_checksum_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'catalogue.json'
            destination.write_bytes(b'local data')
            asset = {'sha256': hashlib.sha256(b'expected').hexdigest(), 'url': 'https://invalid.example'}
            with patch.object(setup.urllib.request, 'urlopen') as network:
                with self.assertRaisesRegex(RuntimeError, 'left untouched'):
                    setup.download(asset, destination)
                network.assert_not_called()
            self.assertEqual(destination.read_bytes(), b'local data')

    def test_incomplete_download_never_becomes_an_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'jdk.tar.gz'
            asset = {'sha256': hashlib.sha256(b'complete').hexdigest(), 'url': 'https://invalid.example'}
            with patch.object(setup.urllib.request, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(b'truncated')), \
                    patch.object(setup.time, 'sleep'):
                with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                    setup.download(asset, destination)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name(destination.name + '.part').exists())

    def test_cached_verified_asset_needs_no_network(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'wheel.whl'
            destination.write_bytes(b'wheel')
            with patch.object(setup.urllib.request, 'urlopen') as network:
                self.assertEqual(setup.download({'sha256': setup.sha256(destination)}, destination), destination)
                network.assert_not_called()

    def test_archive_traversal_and_external_links_are_rejected(self):
        for kind in ('path', 'symlink', 'hardlink'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / 'asset.tar.gz'
                member = tarfile.TarInfo('../escape' if kind == 'path' else 'escape')
                if kind != 'path':
                    member.type = tarfile.SYMTYPE if kind == 'symlink' else tarfile.LNKTYPE
                    member.linkname = '../../escape'
                with tarfile.open(archive, 'w:gz') as tar:
                    tar.addfile(member)
                target = root / 'extract'
                target.mkdir()
                with self.assertRaisesRegex(RuntimeError, 'Unsafe archive'):
                    setup.extract(archive, target)
                self.assertEqual(list(target.iterdir()), [])

    def test_conflicting_data_link_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            link = root / 'data'
            link.symlink_to('previous-dataset')
            with self.assertRaisesRegex(RuntimeError, 'left untouched'):
                setup.link_directory(link, root / 'new-dataset')
            self.assertEqual(link.readlink(), Path('previous-dataset'))

    def test_runtime_patch_mismatch_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ('verl-agent/verl/__init__.py',
                             'verl-agent/' + str(setup.WEBSHOP / 'web_agent_site/engine/engine.py'),
                             'patches/verl-agent/verl/__init__.py'):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('runtime' if relative.startswith('verl-agent') else 'changed patch')
            with patch.object(setup, 'ROOT', root):
                with self.assertRaisesRegex(RuntimeError, 'Runtime overlay differs'):
                    setup.provision_source({}, root / 'cache')
            self.assertEqual((root / 'verl-agent/verl/__init__.py').read_text(), 'runtime')


if __name__ == '__main__':
    unittest.main()
