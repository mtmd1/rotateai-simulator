'''
tests/test_main.py
Unit tests for simulator/__main__.py

Created: 2026-02-26
 Author: Maxence Morel Dierckx
'''
import sys
import signal
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from pathlib import Path
from simulator.__main__ import handle_shutdown, simulate, main, validate_output_path


# MARK: handle_shutdown

class TestHandleShutdown:
    '''Tests for the signal handler.'''

    def test_exits_on_sigint(self):
        with pytest.raises(SystemExit):
            handle_shutdown(signal.SIGINT, None)

    def test_exits_on_sigterm(self):
        with pytest.raises(SystemExit):
            handle_shutdown(signal.SIGTERM, None)

    def test_exit_code_is_one(self):
        with pytest.raises(SystemExit) as exc_info:
            handle_shutdown(signal.SIGINT, None)
        assert exc_info.value.code == 1

    def test_prints_signal_name(self, capsys):
        with pytest.raises(SystemExit):
            handle_shutdown(signal.SIGINT, None)
        assert 'SIGINT' in capsys.readouterr().err

    def test_prints_sigterm_name(self, capsys):
        with pytest.raises(SystemExit):
            handle_shutdown(signal.SIGTERM, None)
        assert 'SIGTERM' in capsys.readouterr().err


# MARK: simulate

class TestSimulate:
    '''Tests for the simulate orchestrator.'''

    def _make_args(self, config='config.toml', binary='path/to/inference', data='data/', output=None):
        return SimpleNamespace(config=config, binary=binary, data=data, output=output)

    @patch('simulator.__main__.save_report')
    @patch('simulator.__main__.Simulator')
    @patch('simulator.__main__.Data')
    @patch('simulator.__main__.Config')
    def test_loads_config_data_simulator(self, MockConfig, MockData, MockSimulator, mock_save):
        MockData.return_value.__iter__ = MagicMock(return_value=iter([]))
        MockData.return_value.__len__ = MagicMock(return_value=0)
        args = self._make_args()
        simulate(args)

        MockConfig.assert_called_once_with('config.toml')
        MockData.assert_called_once_with('data/')
        MockSimulator.assert_called_once_with('path/to/inference')

    @patch('simulator.__main__.save_report')
    @patch('simulator.__main__.Simulator')
    @patch('simulator.__main__.Data')
    @patch('simulator.__main__.Config')
    def test_runs_each_batch(self, MockConfig, MockData, MockSimulator, mock_save):
        batch_0 = {'_source': 'a.mat'}
        batch_1 = {'_source': 'b.mat'}
        MockData.return_value.__iter__ = MagicMock(return_value=iter([batch_0, batch_1]))
        MockData.return_value.__len__ = MagicMock(return_value=2)
        mock_sim = MockSimulator.return_value

        args = self._make_args()
        simulate(args)

        assert mock_sim.run.call_count == 2

    @patch('simulator.__main__.save_report')
    @patch('simulator.__main__.Simulator')
    @patch('simulator.__main__.Data')
    @patch('simulator.__main__.Config')
    def test_saves_report_per_batch(self, MockConfig, MockData, MockSimulator, mock_save):
        batch_0 = {'_source': 'a.mat'}
        batch_1 = {'_source': 'b.mat'}
        MockData.return_value.__iter__ = MagicMock(return_value=iter([batch_0, batch_1]))
        MockData.return_value.__len__ = MagicMock(return_value=2)

        args = self._make_args()
        simulate(args)

        assert mock_save.call_count == 2

    @patch('simulator.__main__.datetime')
    @patch('simulator.__main__.save_report')
    @patch('simulator.__main__.Simulator')
    @patch('simulator.__main__.Data')
    @patch('simulator.__main__.Config')
    def test_report_name_derives_from_binary(self, MockConfig, MockData, MockSimulator, mock_save, mock_dt):
        mock_dt.now.return_value = datetime(2026, 1, 1, 12, 30, 45)
        MockData.return_value.__iter__ = MagicMock(return_value=iter([{'_source': 'data.mat'}]))
        MockData.return_value.__len__ = MagicMock(return_value=1)

        args = self._make_args(binary='some/nested/path/inference')
        simulate(args)

        name_arg = mock_save.call_args_list[0][0][0]
        assert name_arg == 'simreport_inference_data_123045'

    @patch('simulator.__main__.datetime')
    @patch('simulator.__main__.save_report')
    @patch('simulator.__main__.Simulator')
    @patch('simulator.__main__.Data')
    @patch('simulator.__main__.Config')
    def test_report_name_uses_batch_name(self, MockConfig, MockData, MockSimulator, mock_save, mock_dt):
        mock_dt.now.return_value = datetime(2026, 1, 1, 9, 5, 0)
        batches = [
            {'_source': 'mn11_157aprh_tiny.mat'},
            {'_source': 'deploy_test.mat'},
            {'_source': 'simple.mat'},
        ]
        MockData.return_value.__iter__ = MagicMock(return_value=iter(batches))
        MockData.return_value.__len__ = MagicMock(return_value=3)

        args = self._make_args(binary='inference')
        simulate(args)

        names = [c[0][0] for c in mock_save.call_args_list]
        assert names == [
            'simreport_inference_mn11-157aprh-tiny_090500',
            'simreport_inference_deploy-test_090500',
            'simreport_inference_simple_090500',
        ]

    @patch('simulator.__main__.save_report')
    @patch('simulator.__main__.Simulator')
    @patch('simulator.__main__.Data')
    @patch('simulator.__main__.Config')
    def test_prints_done(self, MockConfig, MockData, MockSimulator, mock_save, capsys):
        MockData.return_value.__iter__ = MagicMock(return_value=iter([]))
        MockData.return_value.__len__ = MagicMock(return_value=0)

        simulate(self._make_args())
        assert 'Done.' in capsys.readouterr().out


# MARK: main

class TestMain:
    '''Tests for the CLI entry point.'''

    @patch('simulator.__main__.simulate')
    def test_registers_signal_handlers(self, mock_simulate):
        with patch('simulator.__main__.signal.signal') as mock_signal, \
             patch('simulator.__main__.argparse.ArgumentParser') as MockParser:
            MockParser.return_value.parse_args.return_value = SimpleNamespace(func=mock_simulate)
            main()

        calls = mock_signal.call_args_list
        sig_nums = [c[0][0] for c in calls]
        assert signal.SIGINT in sig_nums
        assert signal.SIGTERM in sig_nums

    @patch('simulator.__main__.simulate')
    def test_requires_config_arg(self, mock_simulate):
        with patch('sys.argv', ['simulate', '--binary', 'b', '--data', 'd']):
            with pytest.raises(SystemExit):
                main()

    @patch('simulator.__main__.simulate')
    def test_requires_binary_arg(self, mock_simulate):
        with patch('sys.argv', ['simulate', '--config', 'c', '--data', 'd']):
            with pytest.raises(SystemExit):
                main()

    @patch('simulator.__main__.simulate')
    def test_requires_data_arg(self, mock_simulate):
        with patch('sys.argv', ['simulate', '--config', 'c', '--binary', 'b']):
            with pytest.raises(SystemExit):
                main()

    @patch('simulator.__main__.simulate')
    def test_dispatches_to_simulate(self, mock_simulate):
        with patch('sys.argv', ['simulate', '-c', 'conf.toml', '-b', 'bin', '-d', 'data/']):
            main()
        mock_simulate.assert_called_once()

    @patch('simulator.__main__.simulate')
    def test_args_passed_correctly(self, mock_simulate):
        with patch('sys.argv', ['simulate', '-c', 'my.toml', '-b', 'my_bin', '-d', 'my_data/']):
            main()
        args = mock_simulate.call_args[0][0]
        assert args.config == 'my.toml'
        assert args.binary == 'my_bin'
        assert args.data == 'my_data/'

    @patch('simulator.__main__.simulate')
    def test_output_defaults_to_none(self, mock_simulate):
        with patch('sys.argv', ['simulate', '-c', 'c', '-b', 'b', '-d', 'd']):
            main()
        args = mock_simulate.call_args[0][0]
        assert args.output is None

    @patch('simulator.__main__.simulate')
    def test_output_arg_passed(self, mock_simulate):
        with patch('sys.argv', ['simulate', '-c', 'c', '-b', 'b', '-d', 'd', '-o', '/tmp/out']):
            main()
        args = mock_simulate.call_args[0][0]
        assert args.output == '/tmp/out'


# MARK: validate_output_path

class TestValidateOutputPath:
    '''Tests for output path validation.'''

    def test_none_returns_cwd(self):
        result = validate_output_path(None)
        assert result == Path.cwd()

    def test_existing_directory_returns_path(self, tmp_path):
        result = validate_output_path(str(tmp_path))
        assert result == tmp_path

    def test_relative_directory_resolved(self, tmp_path, monkeypatch):
        subdir = tmp_path / 'output'
        subdir.mkdir()
        monkeypatch.chdir(tmp_path)
        result = validate_output_path('output')
        assert result == subdir

    def test_file_path_exits(self, tmp_path):
        f = tmp_path / 'file.txt'
        f.write_text('hello')
        with pytest.raises(SystemExit):
            validate_output_path(str(f))

    def test_nonexistent_path_created(self, tmp_path):
        new_dir = tmp_path / 'new_output'
        assert not new_dir.exists()
        result = validate_output_path(str(new_dir))
        assert result == new_dir
        assert new_dir.is_dir()

    def test_nonexistent_nested_path_created(self, tmp_path):
        nested = tmp_path / 'a' / 'b' / 'c'
        result = validate_output_path(str(nested))
        assert result == nested
        assert nested.is_dir()

    def test_mkdir_failure_exits(self, tmp_path):
        bad_path = tmp_path / 'no_such_dir' / 'child'
        with patch('simulator.__main__.Path.mkdir', side_effect=PermissionError('denied')):
            with pytest.raises(SystemExit):
                validate_output_path(str(bad_path))
